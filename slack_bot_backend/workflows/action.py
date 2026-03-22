"""ACTION workflow: runs Aider in non-interactive mode and opens a GitHub PR."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
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
from slack_bot_backend.services.interfaces import LanguageModel, PullRequestDraft, SlackGateway, SupabaseRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-tier mapping — keyed by the complexity_tier returned by the evaluator
# ---------------------------------------------------------------------------

_MODEL_TIER_MAP: dict[str, str] = {
    "simple": "groq/llama-3.3-70b-versatile",
    "complex": "anthropic/claude-sonnet-4-20250514",
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

You have TWO jobs:
  1. Decide whether the developer's request is clear enough for Aider to \
act on. Lean STRONGLY toward "actionable" — Aider is smart and can explore.
  2. If actionable, classify its COMPLEXITY so the system picks the right LLM.

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
OUTPUT FORMAT — respond with ONLY a JSON object, no markdown fences:

{{
  "is_actionable": <true | false>,
  "clarifying_question": "<one focused question>" | null,
  "optimized_prompt": "<rewritten prompt>" | null,
  "complexity_tier": "simple" | "complex"
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

    async def create_pull_request(self, draft: PullRequestDraft) -> str: ...


class ActionWorkflow:
    def __init__(
        self,
        *,
        slack: SlackGateway,
        git: ActionGitOperations,
        llm: LanguageModel,
        repo_path: str,
        github_repository: str = "",
        supabase: SupabaseRepository | None = None,
        model_tier_map: dict[str, str] | None = None,
        aider_bin: str | None = None,
    ) -> None:
        self.slack = slack
        self.git = git
        self.llm = llm
        self.repo_path = repo_path
        self.github_repository = github_repository
        self.supabase = supabase
        self.model_tier_map: dict[str, str] = dict(model_tier_map or _MODEL_TIER_MAP)
        self.aider_bin = aider_bin

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, request: ActionRequest) -> ActionRouteResult:
        try:
            eval_result = await self._evaluate_request(request)
            if not eval_result.is_actionable:
                return await self._handle_needs_clarification(request, eval_result)

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
                request, optimized_prompt=optimized_prompt, model=selected_model,
            )
            if aider_result.returncode != 0:
                if aider_result.test_attempts > 0:
                    return await self._handle_test_failure(request, aider_result)
                return await self._handle_failure(request, aider_result)
            return await self._handle_success(request, aider_result, model=selected_model)
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

    async def _evaluate_request(self, request: ActionRequest) -> EvaluationResult:
        """Call the fast evaluator LLM; fail open (treat as actionable) on any error."""
        prompt = _EVALUATOR_SYSTEM_PROMPT.format(user_request=request.request)
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
        # Fail open: pass the original request straight to Aider (defaults to "simple")
        return EvaluationResult(is_actionable=True, optimized_prompt=request.request)

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
        return env

    _GIT_TIMEOUT = 120  # seconds
    _AIDER_TIMEOUT = 600  # 10 minutes
    _TEST_TIMEOUT = 300  # 5 minutes

    async def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the target repo."""
        logger.debug("Running git %s", " ".join(args))
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
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

    def _detect_test_command(self) -> list[str] | None:
        """Detect the test command for the repository. Returns None if unknown."""
        repo = Path(self.repo_path)
        # Node / npm projects
        if (repo / "package.json").exists():
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
        self, test_cmd: list[str], env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Run the test suite and return the completed process."""
        try:
            return await asyncio.to_thread(
                subprocess.run,
                test_cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
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

    async def _run_aider(
        self,
        request: ActionRequest,
        *,
        optimized_prompt: str,
        model: str,
        existing_branch: str | None = None,
    ) -> AiderResult:
        if not self.repo_path:
            raise RuntimeError(
                "repo_path is empty. Set SLACK_BOT_REPO_PATH to an existing local clone."
            )
        subprocess_env = self._build_subprocess_env()

        # 0. Record the current branch so we can return to it on failure,
        #    stash any dirty working-tree changes.
        head_ref = await self._git("rev-parse", "--abbrev-ref", "HEAD")
        base_branch = head_ref.stdout.strip() or "main"

        stash_result = await self._git("stash", "--include-untracked")
        did_stash = "No local changes" not in (stash_result.stdout or "")

        if existing_branch is not None:
            # ── Existing branch: fetch, checkout, and pull ──
            branch_name = existing_branch
            await self._git("fetch", "origin", existing_branch)
            checkout_proc = await self._git("checkout", existing_branch)
            if checkout_proc.returncode != 0:
                if did_stash:
                    await self._git("stash", "pop")
                return AiderResult(
                    branch_name=branch_name,
                    stdout=checkout_proc.stdout,
                    stderr=checkout_proc.stderr,
                    returncode=checkout_proc.returncode,
                )
            await self._git("pull", "origin", existing_branch)
            # Record the current HEAD so we can detect new commits
            base_sha_proc = await self._git("rev-parse", "HEAD")
            base_sha = base_sha_proc.stdout.strip()
        else:
            # ── New branch: create from origin/main ──
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            branch_name = f"ai-update-{timestamp}"

            # Fetch latest remote state so the branch starts clean.
            await self._git("fetch", "origin", "main")

            # Record the commit SHA we'll branch from (remote main, not local).
            base_sha_proc = await self._git("rev-parse", "origin/main")
            base_sha = base_sha_proc.stdout.strip()

            # 1. Create a new branch from origin/main (ignoring local-only commits)
            checkout_proc = await self._git("checkout", "-b", branch_name, "origin/main")
            if checkout_proc.returncode != 0:
                # Restore stash before returning
                if did_stash:
                    await self._git("stash", "pop")
                return AiderResult(
                    branch_name=branch_name,
                    stdout=checkout_proc.stdout,
                    stderr=checkout_proc.stderr,
                    returncode=checkout_proc.returncode,
                )

        # 2. Run Aider non-interactively with the routed model
        if self.aider_bin:
            aider_bin = self.aider_bin
        else:
            venv_bin = Path(sys.executable).parent
            aider_bin = str(venv_bin / "aider")
        aider_cmd = [
            aider_bin,
            "--model", model,
            "--yes-always",
            "--no-show-model-warnings",
            "--message", optimized_prompt,
        ]
        logger.info(
            "Launching Aider subprocess",
            extra={
                "cwd": self.repo_path,
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
                cwd=self.repo_path,
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
        # Compare current HEAD against the base SHA to see if any new commits
        # were added — this catches Aider auth errors that exit 0 and also
        # prevents pushing dirty working-tree changes that were already present.
        current_sha_proc = await self._git("rev-parse", "HEAD")
        current_sha = current_sha_proc.stdout.strip()
        aider_made_commits = current_sha != base_sha

        if aider_proc.returncode != 0 or not aider_made_commits:
            reason = (
                f"exit {aider_proc.returncode}"
                if aider_proc.returncode != 0
                else "no commits produced"
            )
            logger.warning(
                "Aider run unsuccessful (%s); cleaning up branch",
                reason,
                extra={"branch": branch_name, "channel": request.channel},
            )
            # Clean up: switch back to base branch, delete the AI branch,
            # and restore stashed changes.
            await self._git("checkout", base_branch)
            await self._git("branch", "-D", branch_name)
            if did_stash:
                await self._git("stash", "pop")
            return AiderResult(
                branch_name=branch_name,
                stdout=aider_proc.stdout,
                stderr=aider_proc.stderr + (
                    "\n[bot] Aider made no changes." if not aider_made_commits else ""
                ),
                returncode=aider_proc.returncode or 1,
            )

        # 3. Run test validation loop (if a test framework is detected)
        test_cmd = self._detect_test_command()
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
                test_proc = await self._run_test_command(test_cmd, subprocess_env)
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
                            cwd=self.repo_path,
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
                await self._git("checkout", base_branch)
                await self._git("branch", "-D", branch_name)
                if did_stash:
                    await self._git("stash", "pop")
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

        # 4. Push the branch to origin
        logger.info(
            "Pushing branch to origin",
            extra={"branch": branch_name, "channel": request.channel},
        )
        push_proc = await self._git("push", "--force-with-lease", "origin", branch_name)
        logger.info(
            "Push completed",
            extra={"branch": branch_name, "returncode": push_proc.returncode},
        )
        if push_proc.returncode != 0:
            await self._git("checkout", base_branch)
            if did_stash:
                await self._git("stash", "pop")
            return AiderResult(
                branch_name=branch_name,
                stdout=push_proc.stdout,
                stderr=push_proc.stderr,
                returncode=push_proc.returncode,
            )

        # 5. Return to base branch and restore stash
        await self._git("checkout", base_branch)
        if did_stash:
            await self._git("stash", "pop")

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
        self, request: ActionRequest, aider_result: AiderResult, *, model: str = "",
    ) -> ActionRouteResult:
        branch_name = aider_result.branch_name
        pr_url = await self.git.create_pull_request(
            PullRequestDraft(
                title=f"AI Update: {request.request[:72]}",
                body=self._build_pr_body(request, aider_result),
                branch_name=branch_name,
            )
        )

        # Persist the PR → Slack thread mapping so we can handle PR comments
        await self._save_pr_mapping(
            pr_url=pr_url,
            branch_name=branch_name,
            channel_id=request.channel,
            thread_ts=request.thread_ts,
        )

        # Persist the repo config so the bot remembers the last repo it worked with
        await self._save_repository_config()

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

    async def _save_repository_config(self) -> None:
        """Persist the current repo_path and github_repository to Supabase."""
        if self.supabase is None:
            return
        try:
            await self.supabase.save_repository_config(
                repo_path=self.repo_path,
                github_repository=self.github_repository,
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