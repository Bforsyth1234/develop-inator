"""GitHub-backed repository search and pull request operations for ACTION flows."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from urllib.parse import quote

import logging

import httpx

from slack_bot_backend.models.action import ProposedFileChange, RepositorySearchResult
from slack_bot_backend.services.interfaces import PullRequestDraft

logger = logging.getLogger(__name__)


class GitHubGitService:
    def __init__(
        self,
        *,
        token: str,
        repository: str | None,
        client: httpx.AsyncClient | None = None,
        api_base_url: str = "https://api.github.com",
    ) -> None:
        self.repository = repository
        self._client = client or httpx.AsyncClient(
            base_url=api_base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=20.0,
        )

    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str:
        repository = self._require_repository(repository)
        base_branch = await self._get_default_branch(repository)
        return await self._open_pull_request(repository, draft=draft, base_branch=base_branch)

    async def search_repository(
        self,
        query: str,
        *,
        limit: int = 5,
        repository: str | None = None,
    ) -> list[RepositorySearchResult]:
        resolved_repository = self._require_repository(repository)
        response = await self._client.get(
            "/search/code",
            params={"q": f"repo:{resolved_repository} {query}", "per_page": limit},
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        results: list[RepositorySearchResult] = []
        for item in items:
            path = item["path"]
            results.append(
                RepositorySearchResult(
                    path=path,
                    snippet=await self._read_file_excerpt(resolved_repository, path),
                    url=item.get("html_url"),
                )
            )
        return results

    async def apply_changes_and_open_pull_request(
        self,
        *,
        changes: Sequence[ProposedFileChange],
        draft: PullRequestDraft,
        base_branch: str | None = None,
        repository: str | None = None,
    ) -> str:
        resolved_repository = self._require_repository(repository)
        resolved_base_branch = base_branch or await self._get_default_branch(resolved_repository)
        base_sha = await self._get_branch_sha(resolved_repository, resolved_base_branch)
        await self._create_branch(resolved_repository, draft.branch_name, base_sha)
        for change in changes:
            await self._upsert_file(
                resolved_repository,
                path=change.path,
                content=change.content,
                message=change.summary,
                branch_name=draft.branch_name,
                base_branch=resolved_base_branch,
            )
        return await self._open_pull_request(
            resolved_repository,
            draft=draft,
            base_branch=resolved_base_branch,
        )

    def _require_repository(self, repository: str | None) -> str:
        resolved_repository = repository or self.repository
        if not resolved_repository:
            raise ValueError("GitHub repository is not configured")
        return resolved_repository

    async def _get_default_branch(self, repository: str) -> str:
        response = await self._client.get(f"/repos/{repository}")
        response.raise_for_status()
        return response.json()["default_branch"]

    async def _get_branch_sha(self, repository: str, branch_name: str) -> str:
        response = await self._client.get(f"/repos/{repository}/git/ref/heads/{branch_name}")
        response.raise_for_status()
        return response.json()["object"]["sha"]

    async def _create_branch(self, repository: str, branch_name: str, base_sha: str) -> None:
        response = await self._client.post(
            f"/repos/{repository}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
        response.raise_for_status()

    async def _read_file_excerpt(self, repository: str, path: str) -> str:
        response = await self._client.get(f"/repos/{repository}/contents/{quote(path, safe='/')}")
        response.raise_for_status()
        payload = response.json()
        if payload.get("encoding") != "base64":
            return ""
        decoded = base64.b64decode(payload["content"])
        return decoded.decode("utf-8", errors="ignore")[:1200]

    async def _get_existing_file_sha(
        self,
        repository: str,
        path: str,
        *,
        base_branch: str,
    ) -> str | None:
        response = await self._client.get(
            f"/repos/{repository}/contents/{quote(path, safe='/')}",
            params={"ref": base_branch},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("sha")

    async def _upsert_file(
        self,
        repository: str,
        *,
        path: str,
        content: str,
        message: str,
        branch_name: str,
        base_branch: str,
    ) -> None:
        body: dict[str, str] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch_name,
        }
        existing_sha = await self._get_existing_file_sha(repository, path, base_branch=base_branch)
        if existing_sha:
            body["sha"] = existing_sha
        response = await self._client.put(
            f"/repos/{repository}/contents/{quote(path, safe='/')}",
            json=body,
        )
        response.raise_for_status()

    async def _open_pull_request(
        self,
        repository: str,
        *,
        draft: PullRequestDraft,
        base_branch: str,
    ) -> str:
        response = await self._client.post(
            f"/repos/{repository}/pulls",
            json={
                "title": draft.title,
                "body": draft.body,
                "head": draft.branch_name,
                "base": base_branch,
            },
        )
        # 422 usually means a PR already exists for this head branch.
        # Look it up, update its title/body to reflect the latest request,
        # and return the existing URL instead of crashing.
        if response.status_code == 422:
            existing = await self._find_existing_pr(repository, draft.branch_name)
            if existing:
                pr_url, pr_number = existing
                logger.info(
                    "PR already exists for branch %s (#%s), updating title/body",
                    draft.branch_name, pr_number,
                )
                await self._update_pull_request(
                    repository, pr_number, title=draft.title, body=draft.body,
                )
                return pr_url
            # Not a duplicate-PR 422 — raise the original error.
            response.raise_for_status()
        response.raise_for_status()
        return response.json()["html_url"]

    async def _find_existing_pr(
        self, repository: str, head_branch: str,
    ) -> tuple[str, int] | None:
        """Find an open PR for *head_branch* and return ``(html_url, number)``, or ``None``."""
        owner = repository.split("/")[0] if "/" in repository else ""
        head_param = f"{owner}:{head_branch}" if owner else head_branch
        response = await self._client.get(
            f"/repos/{repository}/pulls",
            params={"head": head_param, "state": "open", "per_page": 1},
        )
        if response.status_code != 200:
            return None
        pulls = response.json()
        if pulls:
            return pulls[0].get("html_url"), pulls[0].get("number")
        return None

    async def _update_pull_request(
        self, repository: str, pr_number: int, *, title: str, body: str,
    ) -> None:
        """PATCH an existing pull request to update its title and body."""
        response = await self._client.patch(
            f"/repos/{repository}/pulls/{pr_number}",
            json={"title": title, "body": body},
        )
        if response.status_code != 200:
            logger.warning(
                "Failed to update PR #%s title/body (status %s)",
                pr_number, response.status_code,
            )

    async def resolve_review_thread(self, pr_url: str, comment_node_id: str) -> None:
        """Resolve a PR review thread via the GitHub GraphQL API.

        Uses the comment's node_id to look up the corresponding review thread,
        then calls the ``resolveReviewThread`` mutation.
        """
        # First, query for the review thread ID that contains this comment
        query = """
        query($nodeId: ID!) {
          node(id: $nodeId) {
            ... on PullRequestReviewComment {
              pullRequestReviewThread {
                id
                isResolved
              }
            }
          }
        }
        """
        resp = await self._client.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"nodeId": comment_node_id}},
        )
        resp.raise_for_status()
        data = resp.json()

        node = (data.get("data") or {}).get("node") or {}
        thread_info = node.get("pullRequestReviewThread")
        if not thread_info:
            logger.warning(
                "Could not find review thread for comment node",
                extra={"comment_node_id": comment_node_id, "response": data},
            )
            return

        if thread_info.get("isResolved"):
            logger.info("Review thread already resolved", extra={"comment_node_id": comment_node_id})
            return

        thread_id = thread_info["id"]

        # Resolve the thread
        mutation = """
        mutation($threadId: ID!) {
          resolveReviewThread(input: {threadId: $threadId}) {
            thread { isResolved }
          }
        }
        """
        resp = await self._client.post(
            "https://api.github.com/graphql",
            json={"query": mutation, "variables": {"threadId": thread_id}},
        )
        resp.raise_for_status()
        result = resp.json()
        errors = result.get("errors")
        if errors:
            logger.warning(
                "GraphQL errors resolving review thread",
                extra={"errors": errors, "thread_id": thread_id},
            )
        else:
            logger.info(
                "Resolved PR review thread",
                extra={"thread_id": thread_id, "comment_node_id": comment_node_id},
            )