"""OpenViking-backed implementation of the ContextSearch and Indexer protocols."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from collections.abc import Sequence

from slack_bot_backend.models.persistence import DocumentationMatch, JSONValue

logger = logging.getLogger(__name__)


class OpenVikingContextService:
    """Wraps the OpenViking ``AsyncHTTPClient`` to satisfy :class:`ContextSearch`.

    The service delegates semantic search to OpenViking's ``find`` endpoint and
    maps the returned ``MatchedContext`` objects back to the application's
    ``DocumentationMatch`` dataclass so that downstream consumers (e.g.
    ``QuestionWorkflow``) remain unchanged.
    """

    def __init__(
        self,
        *,
        openviking_url: str,
        openviking_api_key: str | None = None,
    ) -> None:
        # Lazy-import so the rest of the app doesn't hard-depend on the
        # openviking_cli package when OpenViking is disabled.
        from openviking_cli.client.http import AsyncHTTPClient

        self._client = AsyncHTTPClient(
            base_url=openviking_url,
            api_key=openviking_api_key or "",
        )
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._client.initialize()
            self._initialized = True

    async def match_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        query_text: str = "",
        limit: int = 5,
        min_similarity: float = 0.0,
        metadata_filter: dict[str, JSONValue] | None = None,
    ) -> list[DocumentationMatch]:
        """Perform semantic search via OpenViking ``find``."""
        await self._ensure_initialized()

        # OpenViking's ``find`` uses the raw query text for semantic search.
        # If no query_text was supplied we fall back to an empty search which
        # will return nothing — callers are expected to provide query_text.
        search_text = query_text or ""
        if not search_text:
            logger.warning("OpenViking match_chunks called without query_text; results may be empty")

        try:
            result = await self._client.find(
                query=search_text,
                limit=limit,
                score_threshold=min_similarity if min_similarity > 0.0 else None,
            )
        except Exception:
            logger.exception("OpenViking find() call failed")
            return []

        matches: list[DocumentationMatch] = []
        for ctx in result:
            matches.append(
                DocumentationMatch(
                    source_type=ctx.context_type.value if ctx.context_type else "resource",
                    source_id=ctx.uri,
                    chunk_index=0,
                    title=ctx.uri.rsplit("/", 1)[-1] if ctx.uri else "",
                    path=ctx.uri,
                    content=ctx.overview or ctx.abstract or "",
                    similarity=ctx.score,
                )
            )

        return matches[:limit]

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._initialized:
            await self._client.close()
            self._initialized = False


class OpenVikingIndexer:
    """Indexes a GitHub repository into OpenViking using native ``add_resource``.

    Unlike the Supabase-based :class:`CodebaseIndexer` which manually chunks,
    embeds, and upserts rows, this implementation delegates *all* ingestion
    work to the OpenViking server.  The workflow is:

    1. Shallow-clone the target repository into a temporary directory.
    2. Call ``add_resource(path=<clone_dir>, wait=True)`` so that OpenViking
       processes every file (chunking, embedding, and indexing) server-side.
    3. Return the number of resources reported by OpenViking.
    """

    def __init__(
        self,
        *,
        openviking_url: str,
        openviking_api_key: str | None = None,
        github_token: str = "",
        github_repository: str = "",
        target_uri: str | None = None,
    ) -> None:
        from openviking_cli.client.http import AsyncHTTPClient

        self._client = AsyncHTTPClient(
            url=openviking_url,
            api_key=openviking_api_key or "",
        )
        self._github_token = github_token
        self._github_repository = github_repository
        self._target_uri = target_uri  # e.g. "viking://resources/codebase"
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._client.initialize()
            self._initialized = True

    def _build_clone_url(self) -> str:
        if not self._github_repository:
            raise RuntimeError("github_repository is not set on OpenVikingIndexer.")
        if not self._github_token:
            raise RuntimeError("github_token is not set on OpenVikingIndexer.")
        return (
            f"https://x-access-token:{self._github_token}@github.com/"
            f"{self._github_repository}.git"
        )

    async def reindex(self) -> int:
        """Clone from GitHub and ingest into OpenViking. Returns resource count."""
        await self._ensure_initialized()
        clone_url = self._build_clone_url()

        with tempfile.TemporaryDirectory(prefix="ov-indexer-") as tmp_dir:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", "--depth=1", clone_url, tmp_dir],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                safe_err = (
                    proc.stderr.replace(self._github_token, "***")
                    if self._github_token
                    else proc.stderr
                )
                logger.error("git clone failed: %s", safe_err)
                return 0

            return await self._ingest_directory(tmp_dir)

    async def _ingest_directory(self, repo_path: str) -> int:
        """Upload *repo_path* to OpenViking via ``add_resource``."""
        try:
            result = await self._client.add_resource(
                path=repo_path,
                to=self._target_uri,
                wait=True,
                timeout=300.0,
                ignore_dirs=".git,node_modules,__pycache__,.venv,venv,dist,build,.turbo,.cache,coverage",
            )
            count = result.get("added", 0) if isinstance(result, dict) else 0
            logger.info(
                "OpenViking ingested %d resources from %s",
                count,
                self._github_repository,
            )
            return count
        except Exception:
            logger.exception("OpenViking add_resource failed")
            return 0

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._initialized:
            await self._client.close()
            self._initialized = False

