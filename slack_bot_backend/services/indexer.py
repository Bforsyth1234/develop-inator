"""Codebase indexer: walks a repo, chunks files, embeds, and upserts into Supabase."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import openai

from slack_bot_backend.models.persistence import DocumentationChunkRecord, EmbeddingMetadata
from slack_bot_backend.services.supabase_persistence import (
    AsyncSupabaseTransport,
    DocumentationChunkRepository,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
BATCH_SIZE = 20

INDEXABLE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json", ".css",
    ".html", ".yaml", ".yml", ".toml", ".sql", ".sh", ".env.example",
}

SKIP_DIRS = {
    "node_modules", ".git", ".next", "__pycache__", ".venv", "venv",
    "dist", "build", ".turbo", ".cache", "coverage",
}


def _should_index(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.suffix not in INDEXABLE_EXTENSIONS:
        return False
    if path.stat().st_size > 100_000:
        return False
    return True


def _collect_files(repo_root: Path) -> list[Path]:
    files = sorted(
        p for p in repo_root.rglob("*") if p.is_file() and _should_index(p, repo_root)
    )
    logger.info("Found %d indexable files in %s", len(files), repo_root)
    return files


def _chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def _embed_batch(
    client: openai.AsyncOpenAI, texts: list[str]
) -> list[list[float]]:
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


class CodebaseIndexer:
    """Indexes a local repository into Supabase documentation_chunks for RAG."""

    def __init__(
        self,
        *,
        openai_api_key: str,
        transport: AsyncSupabaseTransport,
        repo_path: str,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._repo = DocumentationChunkRepository(transport)
        self._repo_path = repo_path

    async def reindex(self) -> int:
        """Incrementally index the repo. Returns total chunks upserted."""
        repo_root = Path(self._repo_path).resolve()
        if not repo_root.is_dir():
            logger.error("Repository path does not exist: %s", repo_root)
            return 0

        files = await asyncio.to_thread(_collect_files, repo_root)
        if not files:
            logger.warning("No indexable files found in %s", repo_root)
            return 0

        # 1. Fetch current DB state
        indexed_files = await self._repo.get_indexed_files()
        logger.info("Found %d files currently indexed in Supabase", len(indexed_files))

        # 2. Hash local files
        local_hashes = await asyncio.to_thread(self._hash_files, files, repo_root)

        # 3. Diff
        local_paths = set(local_hashes.keys())
        indexed_paths = set(indexed_files.keys())

        files_to_delete = sorted(indexed_paths - local_paths)
        files_to_index = sorted(
            rel for rel in local_paths
            if rel not in indexed_files or indexed_files[rel] != local_hashes[rel][0]
        )

        logger.info(
            "Incremental diff: %d new/modified, %d deleted, %d unchanged",
            len(files_to_index),
            len(files_to_delete),
            len(local_paths) - len(files_to_index),
        )

        if not files_to_index and not files_to_delete:
            logger.info("Codebase is up to date — nothing to do.")
            return 0

        # 4. Purge old data
        for rel_path in files_to_delete:
            logger.info("Deleting orphaned chunks for: %s", rel_path)
            await self._repo.delete_file_chunks("codebase", rel_path)

        for rel_path in files_to_index:
            if rel_path in indexed_files:
                logger.info("Deleting stale chunks for modified file: %s", rel_path)
                await self._repo.delete_file_chunks("codebase", rel_path)

        if not files_to_index:
            logger.info(
                "No files to re-embed. %d orphaned files cleaned up.",
                len(files_to_delete),
            )
            return 0

        # 5. Chunk and embed new/modified files
        all_chunks: list[tuple[str, str, int, str, str]] = []
        for rel_path in files_to_index:
            file_hash, content = local_hashes[rel_path]
            parts = _chunk_text(content)
            for idx, part in enumerate(parts):
                all_chunks.append((rel_path, rel_path, idx, part, file_hash))

        logger.info("Total chunks to embed: %d", len(all_chunks))

        oai = openai.AsyncOpenAI(api_key=self._openai_api_key)
        total_upserted = 0

        for batch_start in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
            texts = [c[3] for c in batch]
            vectors = await _embed_batch(oai, texts)

            records = []
            for (rel_path, source_id, chunk_idx, text, file_hash), vector in zip(
                batch, vectors
            ):
                chunk_checksum = hashlib.sha256(text.encode()).hexdigest()[:16]
                records.append(
                    DocumentationChunkRecord(
                        source_type="codebase",
                        source_id=source_id,
                        chunk_index=chunk_idx,
                        title=rel_path.split("/")[-1],
                        path=rel_path,
                        content=text,
                        embedding=tuple(vector),
                        metadata={"repo": repo_root.name, "file_checksum": file_hash},
                        embedding_metadata=EmbeddingMetadata(
                            model=EMBEDDING_MODEL,
                            dimensions=len(vector),
                            source_checksum=chunk_checksum,
                        ),
                    )
                )
            await self._repo.upsert_chunks(records)
            total_upserted += len(records)
            logger.info("Upserted %d chunks (total: %d)", len(records), total_upserted)

        logger.info(
            "Incremental indexing complete. %d chunks stored, %d orphaned files removed.",
            total_upserted,
            len(files_to_delete),
        )
        return total_upserted

    @staticmethod
    def _hash_files(
        files: list[Path], repo_root: Path
    ) -> dict[str, tuple[str, str]]:
        """Return ``{rel_path: (file_hash, content)}`` for all readable, non-empty files."""
        result: dict[str, tuple[str, str]] = {}
        for file_path in files:
            rel = str(file_path.relative_to(repo_root))
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                logger.warning("Skipping unreadable file: %s", rel)
                continue
            if not content.strip():
                continue
            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            result[rel] = (file_hash, content)
        return result

