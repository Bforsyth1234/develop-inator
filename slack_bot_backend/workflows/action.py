"""ACTION workflow: runs Aider in non-interactive mode and opens a GitHub PR."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from slack_bot_backend.models.action import (
    ActionRequest,
    ActionRouteResult,
    AiderResult,
    EvaluationResult,
)
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
    ) -> None:
        self.slack = slack
        self.git = git
        self.llm = llm
        self.repo_path = repo_path
        self.github_repository = github_repository
        self.supabase = supabase
        self.model_tier_map: dict[str, str] = dict(model_tier_map or _MODEL_TIER_MAP)

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
            aider_result = await self._run_aider(
                request, optimized_prompt=optimized_prompt, model=selected_model,
            )
            if aider_result.returncode != 0:
                return await self._handle_failure(request, aider_result)
            return await self._handle_success(request, aider_result)
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

    async def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the target repo."""
        return await asyncio.to_thread(
            subprocess.run,
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )

    async def _run_aider(
        self, request: ActionRequest, *, optimized_prompt: str, model: str
    ) -> AiderResult:
        if not self.repo_path:
            raise RuntimeError(
                "repo_path is empty. Set SLACK_BOT_REPO_PATH to an existing local clone."
            )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"ai-update-{timestamp}"
        subprocess_env = self._build_subprocess_env()

        # 0. Record the current branch so we can return to it on failure,
        #    stash any dirty working-tree changes, and ensure we branch
        #    from the *remote* main — not the local HEAD which may have
        #    stale local-only commits.
        head_ref = await self._git("rev-parse", "--abbrev-ref", "HEAD")
        base_branch = head_ref.stdout.strip() or "main"

        stash_result = await self._git("stash", "--include-untracked")
        did_stash = "No local changes" not in (stash_result.stdout or "")

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
        aider_proc: subprocess.CompletedProcess[str] = await asyncio.to_thread(
            subprocess.run,
            aider_cmd,
            capture_output=True,
            text=True,
            cwd=self.repo_path,
            env=subprocess_env,
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

        # 3. Push the branch to origin
        push_proc = await self._git("push", "origin", branch_name)
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

        # 4. Return to base branch and restore stash
        await self._git("checkout", base_branch)
        if did_stash:
            await self._git("stash", "pop")

        return AiderResult(
            branch_name=branch_name,
            stdout=aider_proc.stdout,
            stderr=aider_proc.stderr,
            returncode=0,
        )

    # ------------------------------------------------------------------
    # Result handlers
    # ------------------------------------------------------------------

    async def _handle_success(
        self, request: ActionRequest, aider_result: AiderResult
    ) -> ActionRouteResult:
        branch_name = aider_result.branch_name
        pr_url = await self.git.create_pull_request(
            PullRequestDraft(
                title=f"AI Update: {request.request[:72]}",
                body=self._build_pr_body(request, aider_result),
                branch_name=branch_name,
            )
        )

        # Persist the repo config so the bot remembers the last repo it worked with
        await self._save_repository_config()

        message = (
            f":white_check_mark: Aider pushed `{branch_name}` and opened a PR.\n"
            f"PR: {pr_url}"
        )
        await self._post_message(request, message)
        return ActionRouteResult(
            status="completed",
            provider="aider",
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