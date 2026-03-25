"""ACTION workflow: runs Aider in non-interactive mode and opens a GitHub PR."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from slack_bot_backend.models.action import (
    ActionExecution,
    ActionRequest,
    ActionRouteResult,
    AiderResult,
    EvaluationResult,
)
from slack_bot_backend.models.persistence import ActivePullRequestRecord
from slack_bot_backend.services.interfaces import ContextSearch, LanguageModel, PullRequestDraft, SlackGateway, SupabaseRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-tier mapping — keyed by the complexity_tier returned by the evaluator
# ---------------------------------------------------------------------------

_MODEL_TIER_MAP: dict[str, str] = {
    "simple": "anthropic/claude-sonnet-4-20250514",
    "complex": "anthropic/claude-opus-4-20250514",
}

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
COMPLEXITY TIER — choose EXACTLY one of "simple" or "complex":

  "simple"  — single-component changes, copy/text updates, isolated CSS \
tweaks that likely touch 1-2 files.

  "complex" — multi-file changes like theming overhauls, state management, \
API integrations, routing changes, or anything spanning 3+ files.

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
  "complexity_tier": "simple" | "complex",
  "target_repository": "<repo key from list above>" | null
}}

Rules:
  • If is_actionable is false  → clarifying_question must be non-null; \
optimized_prompt must be null. complexity_tier should be "simple" (ignored).
  • If is_actionable is true   → optimized_prompt must be non-null; \
clarifying_question must be null.
  • optimized_prompt should be a clear instruction to a coding AI. Include \
reasonable defaults when the user didn't specify (e.g. pick good orange and \
purple shades). Tell Aider to explore the codebase to find the right files.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User's request:
{user_request}"""

# ---------------------------------------------------------------------------
# Planner (Expert Architect) prompt — used for complex tasks
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are an Expert Software Architect. You receive a developer's request \
and (optionally) relevant code context retrieved from the repository.

Your job is to produce a *step-by-step markdown implementation spec* that \
a coding agent (Aider) can follow to implement the change safely.

Rules:
  1. Start with a one-line **Summary** of the change.
  2. List every file that must be created or modified, with a brief \
description of what changes in each.
  3. Include code snippets or pseudo-code where helpful.
  4. Call out edge cases, migrations, or test updates needed.
  5. Use markdown formatting (headers, bullet lists, fenced code blocks).
  6. Be precise and concise — the coding agent will follow your spec literally.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Developer's request:
{user_request}

{rag_context}"""


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
        model_tier_map: dict[str, str] | None = None,
        aider_bin: str | None = None,
        context_search: ContextSearch | None = None,
    ) -> None:
        self.slack = slack
        self.git = git
        self.llm = llm
        self.github_token = github_token
        self.repo_map: list[str] = list(repo_map or [])
        self.supabase = supabase
        self.model_tier_map: dict[str, str] = dict(model_tier_map or _MODEL_TIER_MAP)
        self.aider_bin = aider_bin
        self.context_search = context_search
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
            selected_model = self.model_tier_map.get(
                eval_result.complexity_tier,
                self.model_tier_map["simple"],
            )

            # ── Complex task → invoke the Planner instead of Aider ──
            if eval_result.complexity_tier == "complex":
                return await self._handle_complex_plan(
                    request, optimized_prompt=optimized_prompt, model=selected_model,
                )

            # ── Simple task → run Aider directly ──
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
        synthetic_request = ActionRequest(
            channel=channel_id,
            thread_ts=thread_ts,
            request=comment_body,
        )
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

        optimized_prompt = eval_result.optimized_prompt or comment_body
        selected_model = self.model_tier_map.get(
            eval_result.complexity_tier,
            self.model_tier_map["simple"],
        )

        # 4. Complex → generate spec and post for approval
        if eval_result.complexity_tier == "complex":
            await self._handle_complex_plan(
                synthetic_request,
                optimized_prompt=optimized_prompt,
                model=selected_model,
            )
            return

        # 5. Simple → run Aider on existing branch
        #    Acquire a per-branch lock so concurrent review comments for the
        #    same branch are serialized instead of fighting over git locks.
        async with self._branch_lock(branch_name):
            try:
                aider_result = await self._run_aider(
                    synthetic_request,
                    optimized_prompt=optimized_prompt,
                    model=selected_model,
                    existing_branch=branch_name,
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
    # Complex-task planner
    # ------------------------------------------------------------------

    async def _handle_complex_plan(
        self,
        request: ActionRequest,
        *,
        optimized_prompt: str,
        model: str,
    ) -> ActionRouteResult:
        """Invoke the Planner LLM, persist the spec, and post Block Kit buttons."""
        await self._post_message(
            request,
            ":brain: This looks like a complex change — generating an implementation spec…",
        )

        planner_prompt = _PLANNER_SYSTEM_PROMPT.format(
            user_request=optimized_prompt,
            rag_context="",  # TODO: inject RAG context when available
        )
        llm_result = await self.llm.generate(planner_prompt)
        spec = llm_result.content

        # Persist the execution so we can hydrate state on button click
        execution_id = uuid.uuid4().hex
        execution = ActionExecution(
            id=execution_id,
            channel=request.channel,
            thread_ts=request.thread_ts,
            user_id=request.user_id,
            original_request=optimized_prompt,
            generated_spec=spec,
            status="pending",
            model=model,
        )
        if self.supabase is not None:
            try:
                await self.supabase.save_action_execution(execution)
            except Exception:
                logger.warning("Failed to persist action execution", exc_info=True)

        # Post spec + approval buttons via Block Kit
        blocks = self._build_spec_blocks(spec, execution_id)
        try:
            await self.slack.post_blocks(
                request.channel,
                blocks,
                text="Implementation spec ready for review",
                thread_ts=request.thread_ts,
            )
        except Exception:
            logger.exception("Failed to post spec Block Kit message; falling back to plain text")
            await self._post_message(request, f"*Implementation Spec:*\n\n{spec}")

        return ActionRouteResult(
            status="completed",
            provider="planner",
            message="Implementation spec posted for approval.",
        )

    async def generate_revised_spec(
        self,
        *,
        old_spec: str,
        user_feedback: str,
        original_request: str,
    ) -> str:
        """Re-invoke the Planner with prior spec + user feedback to produce V2."""
        prompt = (
            _PLANNER_SYSTEM_PROMPT.format(
                user_request=original_request,
                rag_context="",
            )
            + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Previous spec (V1):\n{old_spec}\n\n"
            f"User feedback:\n{user_feedback}\n\n"
            "Please produce an updated spec that incorporates the feedback."
        )
        llm_result = await self.llm.generate(prompt)
        return llm_result.content

    @staticmethod
    def _split_text_into_chunks(text: str, max_len: int = 2900) -> list[str]:
        """Split *text* into chunks of at most *max_len* characters, breaking at newlines."""
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # Try to break at the last newline before max_len
            cut = text.rfind("\n", 0, max_len)
            if cut <= 0:
                cut = max_len
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    @staticmethod
    def _build_spec_blocks(spec: str, execution_id: str) -> list[dict]:
        """Build Slack Block Kit blocks for the implementation spec with Approve/Reject buttons."""
        blocks: list[dict] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": ":clipboard: *Implementation Spec*"},
            },
        ]
        # Split the spec across multiple section blocks (3000 char limit each)
        for chunk in ActionWorkflow._split_text_into_chunks(spec):
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": chunk},
            })
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":white_check_mark: Approve & Execute"},
                    "style": "primary",
                    "action_id": "approve_spec",
                    "value": execution_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":x: Reject"},
                    "style": "danger",
                    "action_id": "reject_spec",
                    "value": execution_id,
                },
            ],
        })
        return blocks

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

    _GIT_TIMEOUT = 120  # seconds
    _AIDER_TIMEOUT = 600  # 10 minutes
    _TEST_TIMEOUT = 300  # 5 minutes

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
        except subprocess.TimeoutExpired:
            logger.error("Test command timed out after %ds: %s", self._TEST_TIMEOUT, test_cmd)
            return subprocess.CompletedProcess(
                args=test_cmd,
                returncode=1,
                stdout="",
                stderr=f"[bot] Test command timed out after {self._TEST_TIMEOUT}s",
            )

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
        try:
            aider_proc: subprocess.CompletedProcess[str] = await asyncio.to_thread(
                subprocess.run,
                aider_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=subprocess_env,
                timeout=self._AIDER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.error("Aider timed out after %ds", self._AIDER_TIMEOUT)
            aider_proc = subprocess.CompletedProcess(
                args=aider_cmd,
                returncode=1,
                stdout="",
                stderr=f"[bot] Aider timed out after {self._AIDER_TIMEOUT}s",
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
        has_uncommitted = bool(diff_proc.stdout.strip())

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
                    try:
                        await asyncio.to_thread(
                            subprocess.run,
                            retry_cmd,
                            capture_output=True,
                            text=True,
                            cwd=work_dir,
                            env=subprocess_env,
                            timeout=self._AIDER_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired:
                        logger.error("Aider retry timed out after %ds", self._AIDER_TIMEOUT)
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
        pr_url = await self.git.create_pull_request(
            PullRequestDraft(
                title=f"AI Update: {request.request[:72]}",
                body=self._build_pr_body(request, aider_result),
                branch_name=branch_name,
            ),
            repository=target_repo_key,
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

        model_info = f"\n_Model: `{model}`_" if model else ""
        message = (
            f":white_check_mark: Aider pushed `{branch_name}` and opened a PR.\n"
            f"PR: {pr_url}"
            f"{model_info}"
        )
        await self._post_message(request, message)
        return ActionRouteResult(
            status="completed",
            provider=model or "aider",
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