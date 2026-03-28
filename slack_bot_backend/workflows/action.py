"""ACTION workflow: runs Aider in non-interactive mode and opens a GitHub PR."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from slack_bot_backend.models.action import (
    ActionRequest,
    ActionRouteResult,
    AiderResult,
    EvaluationResult,
)
from slack_bot_backend.models.persistence import ActivePullRequestRecord
from slack_bot_backend.services.interfaces import ContextSearch, LanguageModel, PullRequestDraft, SlackGateway, SupabaseRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pre-flight evaluator prompt
# ---------------------------------------------------------------------------

# Sent to the fast evaluator LLM before every Aider execution.
# {user_request} is substituted at call time.
_EVALUATOR_SYSTEM_PROMPT = """\
You are a pre-flight evaluator and model router for an AI coding assistant \
(Aider) that modifies a repository. Aider has FULL access to the \
codebase and can explore files, find variables, locate components, etc.

You have THREE jobs:
  1. Decide whether the developer's request is clear enough for Aider to \
act on. Lean STRONGLY toward "actionable" — Aider is smart and can explore.
  2. If actionable, classify its COMPLEXITY so the system picks the right LLM.
  3. Identify which TARGET REPOSITORY the request is about.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIONABILITY — DEFAULT TO ACTIONABLE. A request is actionable when the \
developer's INTENT is clear, even if they don't specify exact files, hex \
codes, or implementation details. Aider will figure those out.

ACTIONABLE examples (approve these):
  • "Change the app's color scheme to orange and purple" → Aider finds \
theme/CSS variable files and picks reasonable shades.
  • "Add a loading spinner to the dashboard" → Aider locates the dashboard \
component and adds a spinner.
  • "Update the header to show the user's name" → Aider finds the header \
component and the user data source.
  • "Make the sidebar collapsible" → Aider implements a toggle on the sidebar.
  • "Update existing CSS variables" → Aider finds the CSS variable files.

ONLY ask for clarification when the request is genuinely AMBIGUOUS or \
CONTRADICTORY — i.e., you literally cannot determine what the user wants:
  • "Fix it" (fix what?)
  • "Make it better" (better how?)
  • "Update the thing" (which thing?)

When in doubt, mark as ACTIONABLE and let Aider explore the codebase. \
Assume reasonable defaults for colors, styles, and implementations. \
Do NOT ask for hex codes, file paths, or CSS variable names — Aider can \
find those itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEXITY TIER — choose EXACTLY one of "trivial", "standard", or "complex":

  "trivial"  — single-word text changes, typo fixes, or simple config string \
updates that touch exactly one value in one file.

  "standard" — standard single-component changes, isolated bug fixes, copy \
updates, CSS tweaks, or edits that touch 1-2 files.

  "complex"  — ANY of the following:
    • Framework migrations or rewrites (e.g. React → Angular, Express → FastAPI)
    • Large-scale refactors that restructure significant portions of the codebase
    • Multi-file changes like theming overhauls, state management, API \
integrations, routing changes, or anything spanning 3+ files
    • Adding or replacing major libraries/dependencies across the project
    • Architectural changes (e.g. monolith → microservices, REST → GraphQL)
    • Full feature implementations that require new modules, routes, models, \
and tests

When in doubt between "standard" and "complex", choose "complex".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET REPOSITORY — the system manages multiple repositories. You must \
identify which repository the user's request targets. The available \
repositories are:

{available_repos}

ACTIVE THREAD REPOSITORY: {active_repo}

If the user's request does NOT explicitly mention or imply a specific \
repository, you MUST default to the Active Thread Repository shown above \
(unless it says "none").

Pick EXACTLY one key from the list above and include it as \
"target_repository" in your JSON output. If the request does not clearly \
indicate a repository AND no Active Thread Repository is set, set \
"target_repository" to null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — respond with ONLY a JSON object, no markdown fences:

{{
  "is_actionable": <true | false>,
  "clarifying_question": "<one focused question>" | null,
  "optimized_prompt": "<rewritten prompt>" | null,
  "complexity_tier": "trivial" | "standard" | "complex",
  "target_repository": "<repo key from list above>" | null
}}

Rules:
  • If is_actionable is false  → clarifying_question must be non-null; \
optimized_prompt must be null. complexity_tier should be "trivial" (ignored).
  • If is_actionable is true   → optimized_prompt must be non-null; \
clarifying_question must be null.
  • optimized_prompt should be a clear instruction to a coding AI. Include \
reasonable defaults when the user didn't specify (e.g. pick good orange and \
purple shades). Tell Aider to explore the codebase to find the right files.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User's request:
{user_request}"""

# ---------------------------------------------------------------------------
# PR comment classifier prompt
# ---------------------------------------------------------------------------
# Used to decide whether a GitHub PR comment is actionable feedback that
# requires code changes vs. an informational summary (e.g. an automated
# "here's what changed" comment posted when a PR is opened or pushed).

_PR_COMMENT_CLASSIFIER_PROMPT = """\
You are a classifier for GitHub pull-request comments.

Your ONLY job is to decide whether the following comment is:

  "actionable"   — it requests a code change, flags a bug, asks for a fix,
                    suggests an improvement, or asks a question that implies
                    the code should be modified.

  "summary"      — it is an informational summary of the PR changes, an
                    automated description, a CI status report, a review
                    approval/rejection with no specific change request, or
                    any other comment that does NOT ask for a code change.

Respond with ONLY a JSON object (no markdown fences):

{{"classification": "actionable" | "summary"}}

Comment to classify:
{comment_body}"""


class ActionGitOperations(Protocol):
    """Minimal git interface required by the action workflow."""

    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str: ...


class ActionWorkflow:
    def __init__(
        self,
        *,
        slack: SlackGateway,
        git: ActionGitOperations,
        llm: LanguageModel,
        github_token: str = "",
        repo_map: list[str] | None = None,
        supabase: SupabaseRepository | None = None,
        aider_model_trivial: str = "groq/llama-3.3-70b-versatile",
        aider_model_standard: str = "anthropic/claude-sonnet-4-6",
        aider_bin: str | None = None,
        context_search: ContextSearch | None = None,
        openhands_enabled: bool = False,
        openhands_model: str = "anthropic/claude-sonnet-4-6",
        openhands_url: str = "https://app.all-hands.dev",
        openhands_api_key: str | None = None,
    ) -> None:
        self.slack = slack
        self.git = git
        self.llm = llm
        self.github_token = github_token
        self.repo_map: list[str] = list(repo_map or [])
        self.supabase = supabase
        self.aider_model_trivial = aider_model_trivial
        self.aider_model_standard = aider_model_standard
        self.aider_bin = aider_bin
        self.context_search = context_search
        self.openhands_enabled = openhands_enabled
        self.openhands_model = openhands_model
        self.openhands_url = openhands_url.rstrip("/")
        self.openhands_api_key = openhands_api_key
        # Per-branch locks to serialize concurrent Aider runs.
        # When multiple review comments arrive at the same time for the
        # same branch, only one Aider process runs at a time; the rest queue.
        self._branch_locks: dict[str, asyncio.Lock] = {}

    def _branch_lock(self, branch_name: str) -> asyncio.Lock:
        """Return (and lazily create) an ``asyncio.Lock`` for *branch_name*."""
        if branch_name not in self._branch_locks:
            self._branch_locks[branch_name] = asyncio.Lock()
        return self._branch_locks[branch_name]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _resolve_target_repository(self, eval_result: EvaluationResult) -> str | None:
        """Return the repo key if it exists in ``repo_map``, else ``None``."""
        target = eval_result.target_repository
        if target and target in self.repo_map:
            return target
        return None

    def _extract_repo_from_message(self, text: str) -> str | None:
        """Check if the user's message text contains a valid repo from ``repo_map``.

        This handles the case where the user replies with just a repo name
        (e.g. ``Bforsyth1234/rfpinator``) after a "which repository?" clarification.
        """
        stripped = text.strip()
        for repo_key in self.repo_map:
            if repo_key == stripped or repo_key in stripped:
                return repo_key
        return None

    @staticmethod
    def _extract_tier_flag(text: str) -> tuple[str, str | None]:
        """Strip an explicit tier flag from *text* and return ``(cleaned_text, tier)``.

        Users can append ``--trivial``, ``--standard``, or ``--complex`` anywhere
        in their Slack message to bypass the LLM evaluator and force a specific
        routing tier.  The flag is removed from the text before it is passed to
        Aider so the coding agent never sees it.

        Returns a 2-tuple of ``(cleaned_text, forced_tier)`` where *forced_tier*
        is ``None`` when no flag is present.
        """
        import re
        pattern = re.compile(r"\s*--(trivial|standard|complex)\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            return text, None
        tier = match.group(1).lower()
        cleaned = pattern.sub("", text).strip()
        return cleaned, tier

    async def _get_active_thread_repo(self, request: ActionRequest) -> str | None:
        """Fetch the remembered target repository for this thread, if any."""
        if self.supabase is None:
            return None
        try:
            return await self.supabase.get_thread_context(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
            )
        except Exception:
            logger.warning(
                "Failed to read thread context; continuing without it",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
                exc_info=True,
            )
            return None

    async def _save_thread_context(self, request: ActionRequest, target_repository: str) -> None:
        """Persist the resolved target repository for future messages in this thread."""
        if self.supabase is None:
            return
        try:
            await self.supabase.upsert_thread_context(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
                target_repository=target_repository,
            )
        except Exception:
            logger.warning(
                "Failed to save thread context; thread memory will not be available",
                extra={
                    "channel": request.channel,
                    "thread_ts": request.thread_ts,
                    "target_repository": target_repository,
                },
                exc_info=True,
            )

    async def _save_pending_request(self, request: ActionRequest) -> None:
        """Save the original user request so it can be re-executed after repo clarification."""
        if self.supabase is None:
            return
        try:
            await self.supabase.save_pending_request(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
                pending_request=request.request,
            )
        except Exception:
            logger.warning(
                "Failed to save pending request",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
                exc_info=True,
            )

    async def _get_pending_request(self, request: ActionRequest) -> str | None:
        """Retrieve a stored pending request for this thread, if any."""
        if self.supabase is None:
            return None
        try:
            return await self.supabase.get_pending_request(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
            )
        except Exception:
            logger.warning(
                "Failed to read pending request",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
                exc_info=True,
            )
            return None

    async def _clear_pending_request(self, request: ActionRequest) -> None:
        """Clear the stored pending request after successful re-execution."""
        if self.supabase is None:
            return
        try:
            await self.supabase.clear_pending_request(
                channel_id=request.channel,
                thread_ts=request.thread_ts,
            )
        except Exception:
            logger.warning(
                "Failed to clear pending request",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
                exc_info=True,
            )

    async def run(self, request: ActionRequest) -> ActionRouteResult:
        try:
            # ── Pre-flight: fetch remembered thread context ──
            active_thread_repo = await self._get_active_thread_repo(request)
            logger.info(
                "ACTION run: channel=%s thread_ts=%s active_thread_repo=%s repo_map=%s",
                request.channel, request.thread_ts, active_thread_repo, self.repo_map,
            )

            # ── Repo-name reply detection ──
            # If the thread has no saved repo yet, check whether the user's
            # message is simply a repository name (a reply to "which repo?").
            # If so, persist it as the thread context and check for a stored
            # pending request to auto re-execute.
            if active_thread_repo is None and self.repo_map:
                matched_repo = self._extract_repo_from_message(request.request)
                logger.info("Repo-name extraction: request=%r matched=%s", request.request, matched_repo)
                if matched_repo is not None:
                    await self._save_thread_context(request, matched_repo)

                    # Check if there's a pending request to re-execute
                    pending = await self._get_pending_request(request)
                    if pending is not None:
                        logger.info(
                            "Re-executing pending request: %r with repo=%s",
                            pending, matched_repo,
                        )
                        await self._clear_pending_request(request)
                        await self._post_message(
                            request,
                            f":white_check_mark: Got it — targeting `{matched_repo}`. "
                            f"Now executing your original request…",
                        )
                        # Re-invoke run() with the original request and the
                        # now-known thread context.
                        replay_request = ActionRequest(
                            channel=request.channel,
                            thread_ts=request.thread_ts,
                            request=pending,
                        )
                        return await self.run(replay_request)

                    # No pending request — just confirm the repo selection
                    message = (
                        f":white_check_mark: Got it — I'll target `{matched_repo}` "
                        "for this thread. What would you like me to do?"
                    )
                    await self._post_message(request, message)
                    return ActionRouteResult(
                        status="repo_selected",
                        provider="system",
                        message=message,
                    )

            # ── Tier-flag extraction: --trivial / --standard / --complex ──
            cleaned_request, forced_tier = self._extract_tier_flag(request.request)
            if forced_tier is not None:
                request = ActionRequest(
                    channel=request.channel,
                    thread_ts=request.thread_ts,
                    request=cleaned_request,
                    repository=request.repository,
                    base_branch=request.base_branch,
                    user_id=request.user_id,
                )
                logger.info("Tier flag detected: forcing tier=%s", forced_tier)

            eval_result = await self._evaluate_request(
                request, active_thread_repo=active_thread_repo,
            )
            if not eval_result.is_actionable:
                return await self._handle_needs_clarification(request, eval_result)

            # ── Resolve target repository ──
            target_repo_key: str | None = None
            if self.repo_map:
                target_repo_key = self._resolve_target_repository(eval_result)
                # Fall back to the thread-level memory if the evaluator
                # didn't return a valid repository (e.g. user already
                # clarified in a previous message).
                if target_repo_key is None and active_thread_repo is not None:
                    logger.info(
                        "Evaluator did not specify target repo; using thread context: %s",
                        active_thread_repo,
                    )
                    target_repo_key = active_thread_repo
                if target_repo_key is None:
                    # Neither the evaluator nor thread context provided a repo.
                    # Save the original request so we can auto re-execute it
                    # once the user tells us which repo to target.
                    await self._save_pending_request(request)
                    repo_list = ", ".join(f"`{k}`" for k in self.repo_map)
                    message = (
                        ":thinking_face: I couldn't determine which repository "
                        "this change targets. The available repositories are: "
                        f"{repo_list}. Could you clarify which one you mean?"
                    )
                    await self._post_message(request, message)
                    return ActionRouteResult(
                        status="needs_clarification",
                        provider="evaluator",
                        message=message,
                    )

            # ── Persist thread context for future messages ──
            if target_repo_key is not None:
                await self._save_thread_context(request, target_repo_key)

            optimized_prompt = eval_result.optimized_prompt or request.request
            tier = forced_tier or eval_result.complexity_tier
            executor = "OpenHands" if (tier == "complex" and self.openhands_enabled) else "Aider"
            logger.info(
                "Execution plan: tier=%s (forced=%s) → executor=%s",
                tier, forced_tier is not None, executor,
            )

            # ── Complex task → route to OpenHands if enabled ──
            if tier == "complex" and self.openhands_enabled:
                oh_result = await self._run_openhands(
                    request,
                    optimized_prompt=optimized_prompt,
                    openhands_model=self.openhands_model,
                    target_repo_key=target_repo_key,
                )
                if oh_result.returncode != 0:
                    return await self._handle_failure(request, oh_result)
                return await self._handle_success(
                    request, oh_result, model=self.openhands_model, target_repo_key=target_repo_key,
                )

            # ── Standard task (or complex with OpenHands disabled) → Aider + Sonnet ──
            # ── Trivial task → Aider + Groq/Llama ──
            if tier == "trivial":
                selected_model = self.aider_model_trivial
            else:
                selected_model = self.aider_model_standard

            aider_result = await self._run_aider(
                request,
                optimized_prompt=optimized_prompt,
                model=selected_model,
                target_repo_key=target_repo_key,
            )
            if aider_result.returncode != 0:
                if aider_result.test_attempts > 0:
                    return await self._handle_test_failure(request, aider_result)
                return await self._handle_failure(request, aider_result)
            return await self._handle_success(
                request, aider_result, model=selected_model, target_repo_key=target_repo_key,
            )
        except Exception:
            logger.exception(
                "ACTION workflow encountered an unexpected error",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
            )
            message = (
                "I hit an internal error while running the Aider task. "
                "Please review the logs and try again."
            )
            await self._post_message(request, message)
            return ActionRouteResult(status="error", provider="error", message=message)

    # ------------------------------------------------------------------
    # PR comment classifier (summary vs actionable)
    # ------------------------------------------------------------------

    async def _is_actionable_comment(self, comment_body: str) -> bool:
        """Use the LLM to decide if a PR comment is actionable feedback.

        Returns ``True`` when the comment requests a code change and
        ``False`` when it is an informational summary, CI report, or
        approval that does not require action.  Falls back to ``True``
        (fail-open) if the LLM is unavailable or returns unparseable output.
        """
        prompt = _PR_COMMENT_CLASSIFIER_PROMPT.format(comment_body=comment_body)
        try:
            result = await self.llm.generate(prompt)
            text = result.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
                text = "\n".join(inner)
            data = json.loads(text)
            classification = data.get("classification", "actionable")
            logger.info(
                "PR comment classified as %s",
                classification,
                extra={"classification": classification},
            )
            return classification == "actionable"
        except Exception:
            logger.warning(
                "PR comment classifier failed; defaulting to actionable",
                exc_info=True,
            )
            return True

    # ------------------------------------------------------------------
    # PR comment handler (GitHub webhook → Slack → Aider)
    # ------------------------------------------------------------------

    async def handle_pr_comment(
        self,
        *,
        pr_url: str,
        comment_body: str,
        sender: str,
        comment_node_id: str | None = None,
    ) -> None:
        """Orchestrate a response to a GitHub PR comment.

        1. Look up the PR → Slack thread mapping in Supabase.
        2. Post a notification to the original Slack thread.
        3. Evaluate the comment's complexity.
        4. Simple → run Aider on existing branch and push.
        5. Complex → generate a spec and post for approval.
        """
        if self.supabase is None:
            logger.warning("handle_pr_comment called but Supabase is not configured")
            return

        # 1. Look up PR mapping
        mapping = await self.supabase.get_pr_mapping_by_url(pr_url)
        if mapping is None:
            logger.info(
                "No PR mapping found for comment; ignoring",
                extra={"pr_url": pr_url},
            )
            return

        channel_id = mapping.channel_id
        thread_ts = mapping.thread_ts
        branch_name = mapping.branch_name

        # Extract "owner/repo" from the PR URL so we can clone the right repo.
        # PR URLs look like https://github.com/owner/repo/pull/123
        target_repo_key = self._repo_key_from_pr_url(pr_url)

        # Fallback: if the URL didn't parse, try the single-repo shortcut or
        # the thread-level memory (mirrors the fallback logic in run()).
        if target_repo_key is None:
            logger.warning(
                "Could not extract repo key from PR URL; attempting fallback",
                extra={"pr_url": pr_url},
            )
            if len(self.repo_map) == 1:
                target_repo_key = self.repo_map[0]
            else:
                # Try thread context as a last resort
                _thread_repo = await self._get_active_thread_repo(
                    ActionRequest(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        request=comment_body,
                    )
                )
                if _thread_repo is not None:
                    target_repo_key = _thread_repo

        if target_repo_key is None:
            logger.error(
                "Cannot determine target repository for PR comment",
                extra={"pr_url": pr_url},
            )
            try:
                await self.slack.post_message(
                    channel_id,
                    ":x: I couldn't determine which repository this PR belongs to. "
                    "Please check the bot configuration.",
                    thread_ts=thread_ts,
                )
            except Exception:
                logger.warning("Failed to post error to Slack", exc_info=True)
            return

        # 1b. Classify the comment — skip pure summaries / informational posts
        if not await self._is_actionable_comment(comment_body):
            logger.info(
                "PR comment classified as summary; skipping",
                extra={"pr_url": pr_url, "sender": sender},
            )
            return

        # 2. Notify Slack thread
        comment_excerpt = comment_body[:200] + ("…" if len(comment_body) > 200 else "")
        try:
            await self.slack.post_message(
                channel_id,
                f':eyes: I saw a comment from *{sender}* on the PR:\n> "{comment_excerpt}"\n\n'
                "I'm looking into it now…",
                thread_ts=thread_ts,
            )
        except Exception:
            logger.warning("Failed to post PR comment notification to Slack", exc_info=True)

        # 3. Evaluate the comment through the pre-flight evaluator
        cleaned_comment, forced_tier = self._extract_tier_flag(comment_body)
        synthetic_request = ActionRequest(
            channel=channel_id,
            thread_ts=thread_ts,
            request=cleaned_comment,
        )
        if forced_tier is not None:
            logger.info("Tier flag detected in PR comment: forcing tier=%s", forced_tier)

        eval_result = await self._evaluate_request(synthetic_request)

        if not eval_result.is_actionable:
            # Comment is too vague — ask for clarification in the thread
            question = (
                eval_result.clarifying_question
                or "Could you provide more details about what you'd like changed?"
            )
            try:
                await self.slack.post_message(
                    channel_id,
                    f":thinking_face: {question}",
                    thread_ts=thread_ts,
                )
            except Exception:
                logger.warning("Failed to post clarification to Slack", exc_info=True)
            return

        optimized_prompt = eval_result.optimized_prompt or cleaned_comment
        tier = forced_tier or eval_result.complexity_tier
        executor = "OpenHands" if (tier == "complex" and self.openhands_enabled) else "Aider"
        logger.info(
            "Execution plan: tier=%s (forced=%s) → executor=%s",
            tier, forced_tier is not None, executor,
        )

        # 4. Complex → route to OpenHands if enabled
        if tier == "complex" and self.openhands_enabled:
            oh_result = await self._run_openhands(
                synthetic_request,
                optimized_prompt=optimized_prompt,
                openhands_model=self.openhands_model,
                target_repo_key=target_repo_key,
            )
            if oh_result.returncode != 0:
                await self._handle_failure(synthetic_request, oh_result)
            else:
                await self._handle_success(
                    synthetic_request, oh_result,
                    model=self.openhands_model, target_repo_key=target_repo_key,
                )
            return

        # 5. Trivial/Standard → run Aider on existing branch
        #    Acquire a per-branch lock so concurrent review comments for the
        #    same branch are serialized instead of fighting over git locks.
        selected_model = self.aider_model_trivial if tier == "trivial" else self.aider_model_standard
        async with self._branch_lock(branch_name):
            try:
                aider_result = await self._run_aider(
                    synthetic_request,
                    optimized_prompt=optimized_prompt,
                    model=selected_model,
                    existing_branch=branch_name,
                    target_repo_key=target_repo_key,
                )
            except Exception:
                logger.exception("Aider run failed for PR comment feedback")
                try:
                    await self.slack.post_message(
                        channel_id,
                        ":x: I hit an error trying to address the PR comment. Please check the logs.",
                        thread_ts=thread_ts,
                    )
                except Exception:
                    logger.warning("Failed to post error to Slack", exc_info=True)
                return

            if aider_result.returncode != 0:
                stderr_snippet = aider_result.stderr[-500:] or "(empty)"
                try:
                    await self.slack.post_message(
                        channel_id,
                        f":x: Aider couldn't apply the fix (exit {aider_result.returncode}).\n"
                        f"```\n{stderr_snippet}\n```",
                        thread_ts=thread_ts,
                    )
                except Exception:
                    logger.warning("Failed to post failure to Slack", exc_info=True)
                return

            # 6. Resolve the review thread on GitHub if we have a comment node ID
            if comment_node_id:
                try:
                    await self.git.resolve_review_thread(pr_url, comment_node_id)
                except Exception:
                    logger.warning(
                        "Failed to resolve review thread on GitHub",
                        extra={"comment_node_id": comment_node_id},
                        exc_info=True,
                    )

            try:
                await self.slack.post_message(
                    channel_id,
                    ":white_check_mark: I've pushed a fix for your comment to the PR.",
                    thread_ts=thread_ts,
                )
            except Exception:
                logger.warning("Failed to post success to Slack", exc_info=True)

    # ------------------------------------------------------------------
    # Pre-flight evaluator
    # ------------------------------------------------------------------

    async def _evaluate_request(
        self, request: ActionRequest, *, active_thread_repo: str | None = None,
    ) -> EvaluationResult:
        """Call the fast evaluator LLM; fail open (treat as actionable) on any error."""
        repo_keys = list(self.repo_map)
        available_repos = "\n".join(f"  • {k}" for k in repo_keys) if repo_keys else "  (none configured)"
        active_repo_display = active_thread_repo if active_thread_repo else "none"
        prompt = _EVALUATOR_SYSTEM_PROMPT.format(
            user_request=request.request,
            available_repos=available_repos,
            active_repo=active_repo_display,
        )
        try:
            llm_result = await self.llm.generate(prompt)
            parsed = self._parse_evaluation(llm_result.content)
            if parsed is not None:
                return parsed
            logger.warning(
                "Pre-flight evaluator returned invalid JSON; defaulting to actionable",
                extra={"raw": llm_result.content[:300]},
            )
        except Exception:
            logger.warning(
                "Pre-flight evaluator raised an exception; defaulting to actionable",
                exc_info=True,
            )
        # Fail open: pass the original request straight to Aider (defaults to "simple").
        # If there's exactly one repo in the map, default to it so we don't
        # unnecessarily ask the user for clarification.
        fallback_repo = next(iter(self.repo_map), None) if len(self.repo_map) == 1 else None
        return EvaluationResult(
            is_actionable=True,
            optimized_prompt=request.request,
            target_repository=fallback_repo,
        )

    @staticmethod
    def _parse_evaluation(raw: str) -> EvaluationResult | None:
        """Parse the evaluator LLM's JSON output into an EvaluationResult."""
        text = raw.strip()
        # Strip markdown code fences when the model wraps its output
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner)
        try:
            data = json.loads(text)
            return EvaluationResult.model_validate(data)
        except Exception:
            return None

    async def _handle_needs_clarification(
        self, request: ActionRequest, eval_result: EvaluationResult
    ) -> ActionRouteResult:
        question = (
            eval_result.clarifying_question
            or "Could you provide more details about your request?"
        )
        logger.info(
            "Pre-flight evaluator: request needs clarification",
            extra={"channel": request.channel, "thread_ts": request.thread_ts},
        )
        message = (
            ":thinking_face: Before I can make this change, I need a bit more context:\n\n"
            f"{question}"
        )
        await self._post_message(request, message)
        return ActionRouteResult(
            status="needs_clarification",
            provider="evaluator",
            message=message,
        )

    # ------------------------------------------------------------------
    # OpenHands executor (replaces the manual Planner approval flow)
    # ------------------------------------------------------------------

    async def _run_openhands(
        self,
        request: ActionRequest,
        *,
        optimized_prompt: str,
        openhands_model: str,
        target_repo_key: str | None = None,
    ) -> AiderResult:
        """Execute a complex task via the OpenHands local REST API (V0).

        Starts a conversation on a local (or Cloud) OpenHands server using
        the ``/api/conversations`` endpoint, polls for completion, and
        returns an ``AiderResult`` for compatibility with
        ``_handle_success`` / ``_handle_failure``.
        """
        import httpx

        if not self.openhands_api_key:
            logger.error("OpenHands API key is not configured")
            return AiderResult(
                branch_name="",
                stdout="",
                stderr="OpenHands API key is not configured (set SLACK_BOT_OPENHANDS_API_KEY)",
                returncode=1,
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch_name = f"ai-update-{timestamp}"

        await self._post_message(
            request,
            ":brain: This looks like a complex change — running OpenHands agent. "
            "This may take a while (especially on first run). I'll post updates as it progresses.",
        )

        repository = target_repo_key or (self.repo_map[0] if self.repo_map else None)
        if not repository:
            return AiderResult(
                branch_name=branch_name,
                stdout="",
                stderr="No target repository configured for OpenHands",
                returncode=1,
            )

        headers = {
            "X-Session-API-Key": self.openhands_api_key,
            "Content-Type": "application/json",
        }

        # Append explicit push instructions so OpenHands pushes changes
        # back to GitHub on a named branch.  The conversation status
        # endpoint is unreliable (stays RUNNING even after the agent
        # finishes), so we also rely on the events stream to detect
        # completion.
        push_instructions = (
            f"\n\nIMPORTANT — after you have finished making all changes, "
            f"you MUST push them to GitHub so a pull request can be created:\n"
            f"  git checkout -b {branch_name}\n"
            f"  git add -A\n"
            f"  git commit -m \"AI: <short summary of changes>\"\n"
            f"  git push origin {branch_name}\n"
            f"Do NOT skip the push step."
        )

        payload = {
            "initial_user_msg": optimized_prompt + push_instructions,
            "repository": repository,
        }

        if not self.github_token:
            return AiderResult(
                branch_name=branch_name,
                stdout="",
                stderr=(
                    "OpenHands requires a GitHub token to access repositories "
                    "(set SLACK_BOT_GITHUB_TOKEN)"
                ),
                returncode=1,
            )

        try:
            # Two timeout profiles:
            # - short: for quick API calls (token registration, conversation start)
            # - long:  for status polling while the agent works
            short_timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            long_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

            async with httpx.AsyncClient(timeout=short_timeout) as client:
                # 0a. Quick connectivity check — fail fast if the server is
                #     completely unresponsive (e.g. building a Docker image and
                #     blocking all requests).
                try:
                    probe = await client.get(
                        f"{self.openhands_url}/api/options/config",
                        headers=headers,
                    )
                    logger.info(
                        "OpenHands server reachable (HTTP %s)", probe.status_code,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as probe_err:
                    logger.error(
                        "OpenHands server is not responding: %s", probe_err,
                    )
                    return AiderResult(
                        branch_name=branch_name,
                        stdout="",
                        stderr=(
                            f"OpenHands server at {self.openhands_url} is not responding "
                            f"({type(probe_err).__name__}). It may be busy building a "
                            f"sandbox image from a previous task — please try again in "
                            f"a few minutes."
                        ),
                        returncode=1,
                    )

                # 0b. Register the GitHub token so OpenHands can access the repo.
                #     The V0 ``/api/conversations`` endpoint reads provider tokens
                #     from OpenHands' secrets store via ``get_provider_tokens``.
                #     Without this step the server raises:
                #       TypeError: provider_tokens must be a MappingProxyType, got NoneType
                provider_payload = {
                    "provider_tokens": {
                        "github": {
                            "token": self.github_token,
                            "host": "github.com",
                        },
                    },
                }
                token_registered = False
                last_token_error = ""
                for attempt in range(1, 4):  # up to 3 attempts
                    logger.info(
                        "Registering GitHub token with OpenHands (attempt %d)",
                        attempt,
                    )
                    try:
                        token_resp = await client.post(
                            f"{self.openhands_url}/api/add-git-providers",
                            headers=headers,
                            json=provider_payload,
                        )
                        if token_resp.status_code == 200:
                            token_registered = True
                            logger.info("Registered GitHub token with OpenHands successfully")
                            break
                        last_token_error = (
                            f"HTTP {token_resp.status_code}: {token_resp.text}"
                        )
                        logger.warning(
                            "OpenHands add-git-providers attempt %d failed: %s",
                            attempt, last_token_error,
                        )
                    except (httpx.TimeoutException, httpx.NetworkError) as reg_err:
                        last_token_error = f"{type(reg_err).__name__}: {reg_err}"
                        logger.warning(
                            "OpenHands add-git-providers attempt %d timed out: %s",
                            attempt, last_token_error,
                        )
                    if attempt < 3:
                        await asyncio.sleep(15 * attempt)  # 15s, 30s backoff

                if not token_registered:
                    return AiderResult(
                        branch_name=branch_name,
                        stdout="",
                        stderr=(
                            f"Failed to register GitHub token with OpenHands after 3 attempts. "
                            f"Last error: {last_token_error}. "
                            f"The server may be overloaded or the token is invalid."
                        ),
                        returncode=1,
                    )

                # 1. Start the conversation
                resp = await client.post(
                    f"{self.openhands_url}/api/conversations",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code != 200:
                    logger.error(
                        "OpenHands API returned %s: %s",
                        resp.status_code, resp.text,
                    )
                resp.raise_for_status()
                conv_data = resp.json()
                conversation_id = conv_data.get("conversation_id", "")
                logger.info(
                    "OpenHands conversation started: %s", conversation_id,
                )

                # 2. Poll for completion (max ~60 min with 15s intervals)
                #    Switch to the long timeout profile for polling — the
                #    server can be slow to respond while the agent is working.
                client.timeout = long_timeout
                max_polls = 240
                poll_interval = 15
                final_status = "UNKNOWN"
                agent_finished = False
                # Post a Slack update every ~5 minutes so the user knows
                # the agent is still working.
                polls_between_updates = 20  # 20 × 15s = 5 min
                last_reported_status: str | None = None
                # If every poll fails consecutively AND the events stream
                # also can't confirm the agent finished, give up early.
                # When only the metadata file is corrupted (OpenHands bug:
                # JSONDecodeError / 500), the events endpoint is served from
                # a different code path and may still work — so we fall back
                # to it before giving up.
                consecutive_failures = 0
                max_consecutive_failures = 10  # ~2.5 min of solid failures
                # Track which event we've seen so far for pagination.
                last_event_id = 0

                for poll_num in range(1, max_polls + 1):
                    await asyncio.sleep(poll_interval)

                    # Tolerate transient timeouts / errors on status polls.
                    # The OpenHands server may be slow while building sandbox
                    # images or under heavy load; a single failed poll should
                    # not abort the entire conversation.
                    try:
                        status_resp = await client.get(
                            f"{self.openhands_url}/api/conversations/{conversation_id}",
                            headers=headers,
                        )
                        status_resp.raise_for_status()
                        status_data = status_resp.json()
                        final_status = status_data.get("status", "UNKNOWN")
                        consecutive_failures = 0  # reset on success
                    except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as poll_err:
                        consecutive_failures += 1
                        logger.warning(
                            "OpenHands poll %d failed (%d/%d consecutive, will retry): %s",
                            poll_num, consecutive_failures, max_consecutive_failures, poll_err,
                        )
                        # Fallback: when the metadata endpoint fails (e.g. a
                        # corrupted metadata file causes a 500), the events
                        # endpoint is independent and may still be healthy.
                        # Check it now so we don't miss a finished signal.
                        try:
                            events_resp = await client.get(
                                f"{self.openhands_url}/api/conversations/{conversation_id}/events",
                                headers=headers,
                                params={"start_id": last_event_id, "limit": 50},
                            )
                            if events_resp.status_code == 200:
                                fb_events = events_resp.json()
                                if isinstance(fb_events, dict):
                                    fb_events = fb_events.get("events", [])
                                for evt in fb_events:
                                    evt_id = evt.get("id", 0)
                                    if evt_id > last_event_id:
                                        last_event_id = evt_id
                                    extras = evt.get("extras", {})
                                    obs = evt.get("observation", "")
                                    if (
                                        obs == "agent_state_changed"
                                        and extras.get("agent_state") in ("finished", "stopped")
                                    ):
                                        logger.info(
                                            "OpenHands agent state → %s "
                                            "(detected via events despite metadata 500)",
                                            extras["agent_state"],
                                        )
                                        agent_finished = True
                                        break
                        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError):
                            pass  # events also unavailable; rely on consecutive_failures counter

                        if agent_finished:
                            break

                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(
                                "OpenHands conversation %s: %d consecutive poll failures — giving up",
                                conversation_id, consecutive_failures,
                            )
                            return AiderResult(
                                branch_name=branch_name,
                                stdout="",
                                stderr=(
                                    f"OpenHands conversation {conversation_id} is unreachable "
                                    f"({consecutive_failures} consecutive poll failures). "
                                    f"The conversation metadata may be corrupted on the server."
                                ),
                                returncode=1,
                            )
                        continue

                    logger.info(
                        "OpenHands conversation %s status: %s",
                        conversation_id, final_status,
                    )

                    if final_status in ("STOPPED", "ERROR"):
                        break

                    # The conversation status endpoint often stays RUNNING
                    # even after the agent finishes.  Check the events
                    # stream for an authoritative agent_state_changed →
                    # finished/stopped signal.
                    if final_status == "RUNNING":
                        try:
                            events_resp = await client.get(
                                f"{self.openhands_url}/api/conversations/{conversation_id}/events",
                                headers=headers,
                                params={"start_id": last_event_id, "limit": 50},
                            )
                            if events_resp.status_code == 200:
                                events = events_resp.json()
                                if isinstance(events, dict):
                                    events = events.get("events", [])
                                for evt in events:
                                    evt_id = evt.get("id", 0)
                                    if evt_id > last_event_id:
                                        last_event_id = evt_id
                                    extras = evt.get("extras", {})
                                    obs = evt.get("observation", "")
                                    if (
                                        obs == "agent_state_changed"
                                        and extras.get("agent_state") in ("finished", "stopped")
                                    ):
                                        logger.info(
                                            "OpenHands agent state → %s (detected via events)",
                                            extras["agent_state"],
                                        )
                                        agent_finished = True
                                        break
                        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError):
                            pass  # non-critical; we'll retry next cycle

                    if agent_finished:
                        break

                    # Post periodic "still working" updates to Slack
                    if (
                        poll_num % polls_between_updates == 0
                        and final_status != last_reported_status
                    ):
                        elapsed_min = (poll_num * poll_interval) // 60
                        await self._post_message(
                            request,
                            f":hourglass_flowing_sand: OpenHands is still working "
                            f"(status: *{final_status}*, ~{elapsed_min} min elapsed)…",
                        )
                        last_reported_status = final_status

            conversation_link = (
                f"{self.openhands_url}/conversations/{conversation_id}"
            )

            if final_status == "STOPPED" or agent_finished:
                return AiderResult(
                    branch_name=branch_name,
                    stdout=f"OpenHands completed. Conversation: {conversation_link}",
                    stderr="",
                    returncode=0,
                )
            else:
                return AiderResult(
                    branch_name=branch_name,
                    stdout="",
                    stderr=(
                        f"OpenHands conversation ended with status: {final_status}. "
                        f"See: {conversation_link}"
                    ),
                    returncode=1,
                )

        except httpx.TimeoutException as exc:
            logger.exception(
                "OpenHands API call timed out",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
            )
            return AiderResult(
                branch_name=branch_name,
                stdout="",
                stderr=(
                    f"OpenHands request timed out: {exc}. "
                    "If this is the first run, the sandbox image may still be building. "
                    "Please try again in a few minutes."
                ),
                returncode=1,
            )
        except Exception as exc:
            logger.exception(
                "OpenHands API call failed",
                extra={"channel": request.channel, "thread_ts": request.thread_ts},
            )
            return AiderResult(
                branch_name=branch_name,
                stdout="",
                stderr=f"OpenHands execution failed: {type(exc).__name__}: {exc}",
                returncode=1,
            )

    # ------------------------------------------------------------------
    # Aider subprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subprocess_env() -> dict[str, str]:
        """Build a sanitised env dict for Aider, forwarding only the keys it needs."""
        env = os.environ.copy()
        # Forward standard keys
        for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "PATH", "HOME"):
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        # Map our app-level key names to what Aider/litellm expect.
        # SLACK_BOT_LLM_API_KEY holds the Anthropic key in our config.
        llm_key = os.environ.get("SLACK_BOT_LLM_API_KEY")
        if llm_key and not env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = llm_key
        # Map SLACK_BOT_GROQ_API_KEY → GROQ_API_KEY for Aider/litellm.
        groq_key = os.environ.get("SLACK_BOT_GROQ_API_KEY")
        if groq_key and not env.get("GROQ_API_KEY"):
            env["GROQ_API_KEY"] = groq_key
        # Suppress noisy HuggingFace Hub warnings in Aider output.
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        return env

    _GIT_TIMEOUT = 120   # seconds
    _AIDER_TIMEOUT = 1800  # 30 minutes
    _TEST_TIMEOUT = 600   # 10 minutes
    _STALL_SECONDS = 15   # how long to wait with no output before assuming a prompt
    _USER_REPLY_TIMEOUT = 300  # 5 min to wait for the user's Slack reply
    _REPLY_POLL_INTERVAL = 5   # poll Slack every 5s for new messages

    async def _git(self, *args: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
        """Run a git command in an explicit *cwd* (required for stateless mode)."""
        work_dir = cwd or os.getcwd()
        logger.debug("Running git %s (cwd=%s)", " ".join(args), work_dir)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=self._GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.error("git %s timed out after %ds", args[0] if args else "?", self._GIT_TIMEOUT)
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=1,
                stdout="",
                stderr=f"[bot] git {args[0] if args else '?'} timed out after {self._GIT_TIMEOUT}s",
            )
        logger.debug("git %s → exit %d", args[0] if args else "?", result.returncode)
        return result

    # ------------------------------------------------------------------
    # Interactive Aider subprocess
    # ------------------------------------------------------------------

    async def _run_aider_subprocess(
        self,
        cmd: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        channel: str | None = None,
        thread_ts: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an Aider command with interactive stall detection.

        When *channel* and *thread_ts* are provided and the process produces
        no new output for ``_STALL_SECONDS``, the last chunk of output is
        forwarded to the Slack thread as a prompt.  The method then polls
        ``fetch_replies`` for a user response and pipes it into stdin.

        If Slack integration details are not provided, the process runs with
        ``stdin`` closed (equivalent to the old ``subprocess.DEVNULL`` mode).
        """
        interactive = channel is not None and thread_ts is not None

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if interactive else asyncio.subprocess.DEVNULL,
            cwd=cwd,
            env=env,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        last_output_time = asyncio.get_event_loop().time()
        # Track the latest Slack message ts we've already seen so we don't
        # re-read old replies when polling.
        last_seen_ts: str | None = None

        async def _read_stream(
            stream: asyncio.StreamReader | None,
            dest: list[str],
        ) -> None:
            nonlocal last_output_time
            if stream is None:
                return
            while True:
                try:
                    chunk = await asyncio.wait_for(stream.read(4096), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                dest.append(text)
                last_output_time = asyncio.get_event_loop().time()

        async def _monitor() -> None:
            """Detect stalls and relay prompts to Slack."""
            nonlocal last_seen_ts, last_output_time
            if not interactive:
                return
            assert channel is not None and thread_ts is not None  # for type checker
            while proc.returncode is None:
                await asyncio.sleep(1)
                elapsed = asyncio.get_event_loop().time() - last_output_time
                if elapsed < self._STALL_SECONDS:
                    continue

                # Aider has been silent – likely waiting for input.
                recent_output = "".join(stdout_chunks[-5:])[-2000:]
                if not recent_output.strip():
                    continue

                prompt_msg = (
                    ":robot_face: Aider is waiting for input:\n"
                    f"```\n{recent_output}\n```\n"
                    "Reply in this thread to continue, or say `abort` to cancel."
                )
                await self.slack.post_message(channel, prompt_msg, thread_ts=thread_ts)

                # Poll Slack for a reply
                reply_text = await self._wait_for_slack_reply(
                    channel, thread_ts, after_ts=last_seen_ts,
                )
                if reply_text is None or reply_text.strip().lower() == "abort":
                    logger.info("User aborted interactive Aider session")
                    proc.kill()
                    return

                last_seen_ts = str(asyncio.get_event_loop().time())
                # Feed user reply into Aider's stdin
                if proc.stdin is not None:
                    proc.stdin.write((reply_text + "\n").encode())
                    await proc.stdin.drain()
                last_output_time = asyncio.get_event_loop().time()

        # Run readers + monitor concurrently; overall timeout applies.
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(proc.stdout, stdout_chunks),
                    _read_stream(proc.stderr, stderr_chunks),
                    _monitor(),
                ),
                timeout=self._AIDER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Aider timed out after %ds", self._AIDER_TIMEOUT)
            proc.kill()
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"[bot] Aider timed out after {self._AIDER_TIMEOUT}s",
            )

        await proc.wait()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    async def _wait_for_slack_reply(
        self,
        channel: str,
        thread_ts: str,
        *,
        after_ts: str | None = None,
    ) -> str | None:
        """Poll the Slack thread for a new human reply.

        Returns the reply text, or ``None`` if ``_USER_REPLY_TIMEOUT`` elapses.
        """
        deadline = asyncio.get_event_loop().time() + self._USER_REPLY_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            try:
                messages = await self.slack.fetch_replies(
                    channel, thread_ts, oldest=after_ts,
                )
            except Exception as exc:
                # Permanent Slack API errors (e.g. missing_scope, channel_not_found)
                # will never succeed on retry — abort immediately.
                _PERMANENT_ERRORS = ("missing_scope", "channel_not_found", "invalid_auth", "not_authed", "account_inactive")
                exc_str = str(exc)
                if any(err in exc_str for err in _PERMANENT_ERRORS):
                    logger.error("Permanent Slack API error — aborting reply poll: %s", exc)
                    return None
                logger.warning("fetch_replies failed; will retry", exc_info=True)
                await asyncio.sleep(self._REPLY_POLL_INTERVAL)
                continue

            # Look for human messages (no bot_id) that are newer than our prompt
            for msg in reversed(messages):
                if msg.get("bot_id"):
                    continue
                return msg.get("text", "")

            await asyncio.sleep(self._REPLY_POLL_INTERVAL)

        logger.warning("Timed out waiting for Slack reply after %ds", self._USER_REPLY_TIMEOUT)
        return None

    # ------------------------------------------------------------------
    # Test validation helpers
    # ------------------------------------------------------------------

    _MAX_TEST_ATTEMPTS = 3

    def _detect_install_command(self, work_dir: str | None = None) -> list[str] | None:
        """Detect the dependency-install command for the repository."""
        repo = Path(work_dir or os.getcwd())
        if (repo / "pnpm-lock.yaml").exists():
            return ["pnpm", "install", "--frozen-lockfile"]
        if (repo / "yarn.lock").exists():
            return ["yarn", "install", "--frozen-lockfile"]
        if (repo / "package-lock.json").exists() or (repo / "package.json").exists():
            return ["npm", "ci"]
        if (repo / "requirements.txt").exists():
            return ["pip", "install", "-r", "requirements.txt"]
        return None

    _INSTALL_TIMEOUT = 300  # 5 minutes

    async def _install_dependencies(
        self, env: dict[str, str], *, cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Install project dependencies if a known package manager is detected.

        Returns the completed process, or *None* if no install command was detected.
        """
        work_dir = cwd or os.getcwd()
        install_cmd = self._detect_install_command(work_dir)
        if install_cmd is None:
            return None
        logger.info("Installing dependencies: %s (cwd=%s)", install_cmd, work_dir)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                install_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                timeout=self._INSTALL_TIMEOUT,
            )
        except FileNotFoundError:
            logger.warning(
                "Package manager %r not found on PATH; skipping dependency install",
                install_cmd[0],
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("Dependency install timed out after %ds: %s", self._INSTALL_TIMEOUT, install_cmd)
            return subprocess.CompletedProcess(
                args=install_cmd,
                returncode=1,
                stdout="",
                stderr=f"[bot] Dependency install timed out after {self._INSTALL_TIMEOUT}s",
            )
        if result.returncode != 0:
            logger.warning(
                "Dependency install failed (exit %d): %s",
                result.returncode, result.stderr[-500:],
            )
        else:
            logger.info("Dependencies installed successfully")
        return result

    def _detect_test_command(self, work_dir: str | None = None) -> list[str] | None:
        """Detect the test command for the repository. Returns None if unknown."""
        repo = Path(work_dir or os.getcwd())
        # Node / npm projects — prefer the lockfile-specific runner
        if (repo / "package.json").exists():
            if (repo / "pnpm-lock.yaml").exists():
                return ["pnpm", "test"]
            if (repo / "yarn.lock").exists():
                return ["yarn", "test"]
            return ["npm", "test"]
        # Python / pytest projects
        if (repo / "pytest.ini").exists():
            return ["pytest"]
        pyproject = repo / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                if "[tool.pytest" in content:
                    return ["pytest"]
            except OSError:
                pass
        if (repo / "tests").is_dir():
            return ["pytest"]
        return None

    async def _run_test_command(
        self, test_cmd: list[str], env: dict[str, str], *, cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the test suite and return the completed process."""
        work_dir = cwd or os.getcwd()
        try:
            return await asyncio.to_thread(
                subprocess.run,
                test_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                timeout=self._TEST_TIMEOUT,
            )
        except FileNotFoundError:
            logger.warning(
                "Test runner %r not found on PATH; skipping test suite",
                test_cmd[0],
            )
            return subprocess.CompletedProcess(
                args=test_cmd,
                returncode=0,
                stdout="",
                stderr=f"[bot] Test runner '{test_cmd[0]}' not found on PATH; skipped",
            )
        except subprocess.TimeoutExpired:
            logger.error("Test command timed out after %ds: %s", self._TEST_TIMEOUT, test_cmd)
            return subprocess.CompletedProcess(
                args=test_cmd,
                returncode=1,
                stdout="",
                stderr=f"[bot] Test command timed out after {self._TEST_TIMEOUT}s",
            )

    @staticmethod
    def _repo_key_from_pr_url(pr_url: str) -> str | None:
        """Extract ``owner/repo`` from a GitHub PR URL.

        Expected format: ``https://github.com/owner/repo/pull/123``
        Returns ``None`` if the URL doesn't match the expected pattern.
        """
        # e.g. ['', 'owner', 'repo', 'pull', '123']
        parts = pr_url.rstrip("/").split("github.com/")
        if len(parts) < 2:
            return None
        segments = parts[1].split("/")
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return None

    def _build_clone_url(self, repository: str) -> str:
        """Build a GitHub clone URL using the ``x-access-token`` scheme.

        *repository* is the ``owner/repo`` key from ``repo_map``.
        """
        if not repository:
            raise RuntimeError(
                "target repository is not set. Ensure repo_map is configured "
                "and the evaluator identifies a target repository."
            )
        if not self.github_token:
            raise RuntimeError(
                "github_token is not set. Set SLACK_BOT_GITHUB_TOKEN."
            )
        return f"https://x-access-token:{self.github_token}@github.com/{repository}.git"

    async def _run_aider(
        self,
        request: ActionRequest,
        *,
        optimized_prompt: str,
        model: str,
        existing_branch: str | None = None,
        target_repo_key: str | None = None,
    ) -> AiderResult:
        clone_url = self._build_clone_url(repository=target_repo_key)
        subprocess_env = self._build_subprocess_env()

        # ── Workspace isolation: every Aider run happens in an ephemeral
        #    temporary clone so concurrent runs never collide on git state.
        with tempfile.TemporaryDirectory(prefix="aider-workspace-") as tmp_dir:
            return await self._run_aider_in_workspace(
                request,
                work_dir=tmp_dir,
                clone_url=clone_url,
                subprocess_env=subprocess_env,
                optimized_prompt=optimized_prompt,
                model=model,
                existing_branch=existing_branch,
            )

    async def _run_aider_in_workspace(
        self,
        request: ActionRequest,
        *,
        work_dir: str,
        clone_url: str,
        subprocess_env: dict[str, str],
        optimized_prompt: str,
        model: str,
        existing_branch: str | None = None,
    ) -> AiderResult:
        """Execute Aider inside an isolated *work_dir* clone."""

        # 0. Clone the repo from GitHub into the ephemeral workspace.
        #    NOTE: clone_url contains a token – never log it.
        clone_proc = await self._git(
            "clone", "--no-checkout", clone_url, work_dir,
        )
        if clone_proc.returncode != 0:
            # Sanitise stderr/stdout so tokens are never leaked.
            safe_stderr = self._sanitise_output(clone_proc.stderr)
            safe_stdout = self._sanitise_output(clone_proc.stdout)
            return AiderResult(
                branch_name="",
                stdout=safe_stdout,
                stderr=safe_stderr,
                returncode=clone_proc.returncode,
            )

        # Configure git identity in the ephemeral clone so Aider can commit.
        await self._git("config", "user.name", "develop-inator[bot]", cwd=work_dir)
        await self._git("config", "user.email", "develop-inator[bot]@users.noreply.github.com", cwd=work_dir)

        # Fetch latest state inside the clone.
        await self._git("fetch", "origin", cwd=work_dir)

        if existing_branch is not None:
            # ── Existing branch: fetch, checkout, and pull ──
            branch_name = existing_branch
            await self._git("fetch", "origin", existing_branch, cwd=work_dir)
            checkout_proc = await self._git("checkout", existing_branch, cwd=work_dir)
            if checkout_proc.returncode != 0:
                return AiderResult(
                    branch_name=branch_name,
                    stdout=checkout_proc.stdout,
                    stderr=checkout_proc.stderr,
                    returncode=checkout_proc.returncode,
                )
            await self._git("pull", "origin", existing_branch, cwd=work_dir)
            base_sha_proc = await self._git("rev-parse", "HEAD", cwd=work_dir)
            base_sha = base_sha_proc.stdout.strip()
        else:
            # ── New branch: create from origin/main ──
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            branch_name = f"ai-update-{timestamp}"

            await self._git("fetch", "origin", "main", cwd=work_dir)

            base_sha_proc = await self._git("rev-parse", "origin/main", cwd=work_dir)
            base_sha = base_sha_proc.stdout.strip()

            checkout_proc = await self._git(
                "checkout", "-b", branch_name, "origin/main", cwd=work_dir,
            )
            if checkout_proc.returncode != 0:
                return AiderResult(
                    branch_name=branch_name,
                    stdout=checkout_proc.stdout,
                    stderr=checkout_proc.stderr,
                    returncode=checkout_proc.returncode,
                )

        # 2. Discover relevant files via context search (OpenViking)
        file_args: list[str] = []
        if self.context_search is not None:
            try:
                matches = await self.context_search.match_chunks(
                    [],
                    query_text=optimized_prompt,
                    limit=8,
                )
                for m in matches:
                    # Extract relative file path from the URI or path field
                    raw = m.path or ""
                    # viking://resources/owner/repo/path/to/file → path/to/file
                    parts = raw.split("/", 5)  # ['viking:', '', 'resources', 'owner', 'repo', 'rest']
                    rel = parts[5] if len(parts) > 5 else raw.rsplit("/", 1)[-1]
                    if rel:
                        full = os.path.join(work_dir, rel)
                        if os.path.isfile(full):
                            file_args.append(rel)
                logger.info(
                    "Context search found %d candidate files, %d exist on disk",
                    len(matches),
                    len(file_args),
                )
            except Exception:
                logger.warning("Context search failed; Aider will use repo map only", exc_info=True)

        # 2b. Run Aider non-interactively with the routed model
        if self.aider_bin:
            aider_bin = self.aider_bin
        else:
            venv_bin = Path(sys.executable).parent
            aider_bin = str(venv_bin / "aider")
        aider_cmd = [
            aider_bin,
            "--model", model,
            "--yes-always",
            "--no-auto-commits",
            "--no-show-model-warnings",
            "--message", optimized_prompt,
            *file_args,
        ]
        logger.info(
            "Launching Aider subprocess",
            extra={
                "cwd": work_dir,
                "branch": branch_name,
                "model": model,
                "channel": request.channel,
            },
        )
        aider_proc = await self._run_aider_subprocess(
            aider_cmd,
            cwd=work_dir,
            env=subprocess_env,
            channel=request.channel,
            thread_ts=request.thread_ts,
        )

        # 2b. Check if Aider failed or made no changes.
        logger.info(
            "Aider subprocess finished (exit %d)\nstdout: %s\nstderr: %s",
            aider_proc.returncode,
            self._sanitise_output(aider_proc.stdout[-3000:]) if aider_proc.stdout else "(empty)",
            self._sanitise_output(aider_proc.stderr[-3000:]) if aider_proc.stderr else "(empty)",
        )
        # With --no-auto-commits, Aider leaves changes uncommitted.
        # Check for either new commits OR a dirty working tree.
        current_sha_proc = await self._git("rev-parse", "HEAD", cwd=work_dir)
        current_sha = current_sha_proc.stdout.strip()
        aider_made_commits = current_sha != base_sha

        # Check for uncommitted changes (staged or unstaged)
        diff_proc = await self._git("status", "--porcelain", cwd=work_dir)
        changed_files = [
            line.strip().split(maxsplit=1)[-1]
            for line in diff_proc.stdout.strip().splitlines()
            if line.strip()
        ]
        # Aider always creates/modifies .gitignore and .aider* housekeeping
        # files.  Ignore these when deciding whether real work was done.
        _AIDER_HOUSEKEEPING = {".gitignore", ".aider.tags.cache.v0/cache.db"}
        real_changes = [
            f for f in changed_files
            if f not in _AIDER_HOUSEKEEPING and not f.startswith(".aider")
        ]
        has_uncommitted = bool(real_changes)

        if aider_proc.returncode != 0 or (not aider_made_commits and not has_uncommitted):
            reason = (
                f"exit {aider_proc.returncode}"
                if aider_proc.returncode != 0
                else "no changes produced"
            )
            logger.warning(
                "Aider run unsuccessful (%s); ephemeral workspace will be cleaned up",
                reason,
                extra={"branch": branch_name, "channel": request.channel},
            )
            return AiderResult(
                branch_name=branch_name,
                stdout=aider_proc.stdout,
                stderr=aider_proc.stderr + (
                    "\n[bot] Aider made no changes." if not aider_made_commits and not has_uncommitted else ""
                ),
                returncode=aider_proc.returncode or 1,
            )

        # Commit any uncommitted changes left by Aider (--no-auto-commits mode)
        if has_uncommitted:
            await self._git("add", "-A", cwd=work_dir)
            await self._git(
                "commit", "-m", f"feat: {optimized_prompt[:72]}",
                cwd=work_dir,
            )
            logger.info("Committed Aider's uncommitted changes", extra={"branch": branch_name})

        # 3. Install dependencies (so test runners like jest/pnpm are available)
        await self._install_dependencies(subprocess_env, cwd=work_dir)

        # 4. Run test validation loop (if a test framework is detected)
        test_cmd = self._detect_test_command(work_dir)
        test_attempts = 0
        last_test_output = ""
        if test_cmd is not None:
            for attempt in range(self._MAX_TEST_ATTEMPTS):
                test_attempts = attempt + 1
                logger.info(
                    "Running test suite (attempt %d/%d)",
                    test_attempts,
                    self._MAX_TEST_ATTEMPTS,
                    extra={"branch": branch_name, "channel": request.channel},
                )
                test_proc = await self._run_test_command(
                    test_cmd, subprocess_env, cwd=work_dir,
                )
                last_test_output = (
                    (test_proc.stderr or "")
                    + "\n"
                    + (test_proc.stdout or "")[-2000:]
                )
                if test_proc.returncode == 0:
                    logger.info(
                        "Tests passed on attempt %d", test_attempts,
                        extra={"branch": branch_name, "channel": request.channel},
                    )
                    break

                logger.warning(
                    "Tests failed on attempt %d/%d",
                    test_attempts,
                    self._MAX_TEST_ATTEMPTS,
                    extra={"branch": branch_name, "channel": request.channel},
                )

                # If we still have retries left, feed failure back to Aider
                if test_attempts < self._MAX_TEST_ATTEMPTS:
                    fix_prompt = (
                        "The tests failed with the following output. "
                        "Please fix the failing tests:\n\n"
                        f"```\n{last_test_output}\n```"
                    )
                    retry_cmd = [
                        aider_bin,
                        "--model", model,
                        "--yes-always",
                        "--no-show-model-warnings",
                        "--message", fix_prompt,
                    ]
                    await self._run_aider_subprocess(
                        retry_cmd,
                        cwd=work_dir,
                        env=subprocess_env,
                        channel=request.channel,
                        thread_ts=request.thread_ts,
                    )
            else:
                # All attempts exhausted — tests still failing
                logger.error(
                    "Tests failed after %d attempts; aborting",
                    self._MAX_TEST_ATTEMPTS,
                    extra={"branch": branch_name, "channel": request.channel},
                )
                return AiderResult(
                    branch_name=branch_name,
                    stdout=aider_proc.stdout,
                    stderr=(
                        f"[bot] Tests failed after {self._MAX_TEST_ATTEMPTS} "
                        f"attempts.\n{last_test_output}"
                    ),
                    returncode=1,
                    test_attempts=test_attempts,
                    test_output=last_test_output,
                )

        # 4. Push the branch to origin (from the ephemeral clone)
        logger.info(
            "Pushing branch to origin",
            extra={"branch": branch_name, "channel": request.channel},
        )
        push_proc = await self._git(
            "push", "--force-with-lease", "origin", branch_name, cwd=work_dir,
        )
        logger.info(
            "Push completed",
            extra={"branch": branch_name, "returncode": push_proc.returncode},
        )
        if push_proc.returncode != 0:
            return AiderResult(
                branch_name=branch_name,
                stdout=push_proc.stdout,
                stderr=push_proc.stderr,
                returncode=push_proc.returncode,
            )

        # 5. Temp directory will be cleaned up automatically on exit.
        return AiderResult(
            branch_name=branch_name,
            stdout=aider_proc.stdout,
            stderr=aider_proc.stderr,
            returncode=0,
            test_attempts=test_attempts,
            test_output=last_test_output,
        )

    # ------------------------------------------------------------------
    # Result handlers
    # ------------------------------------------------------------------

    async def _handle_success(
        self,
        request: ActionRequest,
        aider_result: AiderResult,
        *,
        model: str = "",
        target_repo_key: str | None = None,
    ) -> ActionRouteResult:
        branch_name = aider_result.branch_name
        try:
            pr_url = await self.git.create_pull_request(
                PullRequestDraft(
                    title=f"AI Update: {request.request[:72]}",
                    body=self._build_pr_body(request, aider_result),
                    branch_name=branch_name,
                ),
                repository=target_repo_key,
            )
        except Exception as pr_err:
            logger.error(
                "Failed to create PR for branch %s: %s",
                branch_name, pr_err,
            )
            # The agent may not have pushed the branch.  Inform the user
            # so they can investigate the OpenHands conversation directly.
            provider_label = model or "agent"
            await self._post_message(
                request,
                f":warning: {provider_label} completed the task but I couldn't "
                f"create a PR for branch `{branch_name}`. "
                f"The agent may not have pushed the branch to GitHub.\n"
                f"Error: {pr_err}",
            )
            return ActionRouteResult(
                status="error",
                provider=model or "aider",
                message=f"PR creation failed: {pr_err}",
                branch_name=branch_name,
            )

        # Persist the PR → Slack thread mapping so we can handle PR comments
        await self._save_pr_mapping(
            pr_url=pr_url,
            branch_name=branch_name,
            channel_id=request.channel,
            thread_ts=request.thread_ts,
        )

        # Persist the repo config so the bot remembers the last repo it worked with
        if target_repo_key:
            await self._save_repository_config(repository=target_repo_key)

        provider_label = model or "aider"
        model_info = f"\n_Model: `{model}`_" if model else ""
        message = (
            f":white_check_mark: {provider_label} pushed `{branch_name}` and opened a PR.\n"
            f"PR: {pr_url}"
            f"{model_info}"
        )
        await self._post_message(request, message)
        return ActionRouteResult(
            status="completed",
            provider=provider_label,
            message=message,
            pr_url=pr_url,
            branch_name=branch_name,
        )

    async def _handle_failure(
        self, request: ActionRequest, aider_result: AiderResult
    ) -> ActionRouteResult:
        logger.error(
            "Aider subprocess exited with non-zero status",
            extra={
                "returncode": aider_result.returncode,
                "stderr_head": aider_result.stderr[:500],
                "channel": request.channel,
                "thread_ts": request.thread_ts,
            },
        )
        stderr_snippet = aider_result.stderr[-1500:] or "(empty)"
        stdout_snippet = aider_result.stdout[-800:] or "(empty)"
        message = (
            f":x: Aider failed (exit {aider_result.returncode}).\n\n"
            f"*stderr:*\n```\n{stderr_snippet}\n```\n\n"
            f"*stdout:*\n```\n{stdout_snippet}\n```"
        )
        await self._post_message(request, message)
        return ActionRouteResult(status="error", provider="aider", message=message)

    async def _handle_test_failure(
        self, request: ActionRequest, aider_result: AiderResult
    ) -> ActionRouteResult:
        logger.error(
            "Tests failed after %d attempts",
            aider_result.test_attempts,
            extra={
                "branch": aider_result.branch_name,
                "channel": request.channel,
                "thread_ts": request.thread_ts,
            },
        )
        test_snippet = aider_result.test_output[-1500:] or "(empty)"
        message = (
            f":x: Aider made changes but the tests failed after "
            f"{aider_result.test_attempts} attempts.\n\n"
            f"*Test output:*\n```\n{test_snippet}\n```"
        )
        await self._post_message(request, message)
        return ActionRouteResult(status="error", provider="aider", message=message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pr_body(request: ActionRequest, aider_result: AiderResult) -> str:
        stdout_excerpt = aider_result.stdout[-3000:] if aider_result.stdout else "(no output)"
        return (
            "## AI-generated change\n\n"
            f"**Request:** {request.request}\n\n"
            "**Generated by:** Aider (non-interactive mode)\n\n"
            "<details><summary>Aider output</summary>\n\n"
            f"```\n{stdout_excerpt}\n```\n\n"
            "</details>"
        )

    async def _save_pr_mapping(
        self,
        *,
        pr_url: str,
        branch_name: str,
        channel_id: str,
        thread_ts: str,
    ) -> None:
        """Persist PR → Slack thread mapping to Supabase."""
        if self.supabase is None:
            return
        try:
            await self.supabase.save_pr_mapping(
                ActivePullRequestRecord(
                    pr_url=pr_url,
                    branch_name=branch_name,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                )
            )
        except Exception:
            logger.warning(
                "Failed to persist PR mapping to Supabase; continuing",
                exc_info=True,
            )

    def _sanitise_output(self, text: str) -> str:
        """Strip the GitHub token from any subprocess output to prevent leaks."""
        if self.github_token and self.github_token in text:
            return text.replace(self.github_token, "***")
        return text

    async def _save_repository_config(self, *, repository: str) -> None:
        """Persist the target repository to Supabase."""
        if self.supabase is None:
            return
        try:
            await self.supabase.save_repository_config(
                github_repository=repository,
            )
        except Exception:
            logger.warning(
                "Failed to persist repository config to Supabase; continuing",
                exc_info=True,
            )

    async def _post_message(self, request: ActionRequest, text: str) -> None:
        try:
            await self.slack.post_message(request.channel, text, thread_ts=request.thread_ts)
        except Exception:
            logger.exception("Failed to post ACTION response to Slack")