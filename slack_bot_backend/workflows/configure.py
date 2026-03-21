"""CONFIGURE workflow: extract and persist repository configuration from chat."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from slack_bot_backend.services.interfaces import LanguageModel, SlackGateway, SupabaseRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigureResult:
    """Outcome of a CONFIGURE workflow run."""

    status: str  # "updated", "incomplete", "error"
    message: str
    repo_path: str | None = None
    github_repository: str | None = None


_EXTRACTION_PROMPT = """\
You are a configuration assistant for a Slack bot that works with GitHub repositories.

The user wants to change which repository the bot is working with.
Extract the following values from the user's message:
- repo_path: the local filesystem path to the repository (e.g. "/home/user/projects/my-app")
- github_repository: the GitHub owner/repo identifier (e.g. "octocat/hello-world")

Current configuration:
- repo_path: {current_repo_path}
- github_repository: {current_github_repository}

Return JSON only with this exact schema:
{{"repo_path": "<extracted path or null>", "github_repository": "<extracted owner/repo or null>"}}

If the user only provides one value, set the other to null.
If you cannot extract either value, set both to null.

User's message:
{user_message}"""


class ConfigureWorkflow:
    def __init__(
        self,
        *,
        slack: SlackGateway,
        supabase: SupabaseRepository,
        llm: LanguageModel,
        repo_path: str = "",
        github_repository: str = "",
    ) -> None:
        self.slack = slack
        self.supabase = supabase
        self.llm = llm
        self.repo_path = repo_path
        self.github_repository = github_repository

    async def run(
        self, *, channel: str, thread_ts: str, user_message: str, user_id: str | None = None,
    ) -> ConfigureResult:
        try:
            extracted = await self._extract_config(user_message)
            new_repo_path = extracted.get("repo_path") or None
            new_github_repo = extracted.get("github_repository") or None

            if not new_repo_path and not new_github_repo:
                message = (
                    "I'd like to help you change the repository configuration. "
                    "Please provide at least one of the following:\n"
                    "• *repo_path* — the local filesystem path to the repository\n"
                    "• *github_repository* — the GitHub owner/repo (e.g. `octocat/hello-world`)\n\n"
                    f"Current config:\n"
                    f"• repo_path: `{self.repo_path or '(not set)'}`\n"
                    f"• github_repository: `{self.github_repository or '(not set)'}`"
                )
                await self._post(channel, message, thread_ts)
                return ConfigureResult(status="incomplete", message=message)

            final_repo_path = new_repo_path or self.repo_path
            final_github_repo = new_github_repo or self.github_repository

            await self.supabase.save_repository_config(
                repo_path=final_repo_path,
                github_repository=final_github_repo,
            )

            self.repo_path = final_repo_path
            self.github_repository = final_github_repo

            changes: list[str] = []
            if new_repo_path:
                changes.append(f"• repo_path → `{new_repo_path}`")
            if new_github_repo:
                changes.append(f"• github_repository → `{new_github_repo}`")

            message = (
                ":white_check_mark: Repository configuration updated!\n"
                + "\n".join(changes)
                + "\n\nThese changes take effect immediately for new tasks."
            )
            await self._post(channel, message, thread_ts)
            return ConfigureResult(
                status="updated",
                message=message,
                repo_path=final_repo_path,
                github_repository=final_github_repo,
            )
        except Exception:
            logger.exception("CONFIGURE workflow failed")
            error_msg = "I hit an error while updating the repository configuration. Please try again."
            try:
                await self._post(channel, error_msg, thread_ts)
            except Exception:
                logger.exception("Failed to post CONFIGURE error to Slack")
            return ConfigureResult(status="error", message=error_msg)

    async def _extract_config(self, user_message: str) -> dict[str, str | None]:
        prompt = _EXTRACTION_PROMPT.format(
            current_repo_path=self.repo_path or "(not set)",
            current_github_repository=self.github_repository or "(not set)",
            user_message=user_message,
        )
        result = await self.llm.generate(prompt)
        return self._parse_extraction(result.content)

    @staticmethod
    def _parse_extraction(raw: str) -> dict[str, str | None]:
        candidate = raw.strip()
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {"repo_path": None, "github_repository": None}
            candidate = candidate[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {"repo_path": None, "github_repository": None}
        return {
            "repo_path": payload.get("repo_path"),
            "github_repository": payload.get("github_repository"),
        }

    async def _post(self, channel: str, text: str, thread_ts: str | None) -> None:
        await self.slack.post_message(channel, text, thread_ts=thread_ts)

