#!/usr/bin/env python3
"""Incrementally index a local repository into Supabase documentation_chunks.

Only new or modified files are embedded; deleted files have their chunks purged.

Usage:
    set -a && source .env && set +a
    .venv/bin/python scripts/index_codebase.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

import openai

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slack_bot_backend.models.persistence import DocumentationChunkRecord, EmbeddingMetadata
from slack_bot_backend.services.indexer import (
    BATCH_SIZE,
    EMBEDDING_MODEL,
    _chunk_text,
    _collect_files,
    _embed_batch,
)
from slack_bot_backend.services.supabase_persistence import (
    DocumentationChunkRepository,
    UrllibSupabaseTransport,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_PATH = os.environ.get("SLACK_BOT_REPO_PATH", "")
SUPABASE_URL = os.environ.get("SLACK_BOT_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SLACK_BOT_SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.environ.get("SLACK_BOT_OPENAI_API_KEY", "")


async def main() -> None:
    if not REPO_PATH:
        sys.exit("SLACK_BOT_REPO_PATH is not set")
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("SLACK_BOT_SUPABASE_URL and SLACK_BOT_SUPABASE_SERVICE_ROLE_KEY are required")
    if not OPENAI_API_KEY:
        sys.exit("SLACK_BOT_OPENAI_API_KEY is required")

    repo_root = Path(REPO_PATH).resolve()
    if not repo_root.is_dir():
        sys.exit(f"Repository path does not exist: {repo_root}")

    transport = UrllibSupabaseTransport(base_url=SUPABASE_URL, service_role_key=SUPABASE_KEY)
    repo = DocumentationChunkRepository(transport)

    # ----- 1. Fetch current DB state -----
    logger.info("Fetching indexed file checksums from Supabase...")
    indexed_files = await repo.get_indexed_files()
    logger.info("Found %d files currently indexed in Supabase", len(indexed_files))

    # ----- 2. Hash local files -----
    files = _collect_files(repo_root)
    if not files:
        sys.exit("No indexable files found")

    local_hashes: dict[str, tuple[str, str]] = {}  # rel_path -> (file_hash, content)
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
        local_hashes[rel] = (file_hash, content)

    # ----- 3. Diff the codebase -----
    local_paths = set(local_hashes.keys())
    indexed_paths = set(indexed_files.keys())

    files_to_delete = sorted(indexed_paths - local_paths)
    files_to_index = sorted(
        rel for rel in local_paths
        if rel not in indexed_files or indexed_files[rel] != local_hashes[rel][0]
    )
    unchanged = len(local_paths) - len(files_to_index)

    logger.info(
        "Incremental diff: %d new/modified, %d deleted, %d unchanged",
        len(files_to_index),
        len(files_to_delete),
        unchanged,
    )

    if not files_to_index and not files_to_delete:
        logger.info("Codebase is up to date — nothing to do.")
        return

    # ----- 4. Purge old data -----
    for rel_path in files_to_delete:
        logger.info("Deleting orphaned chunks for: %s", rel_path)
        await repo.delete_file_chunks("codebase", rel_path)

    for rel_path in files_to_index:
        if rel_path in indexed_files:
            logger.info("Deleting stale chunks for modified file: %s", rel_path)
            await repo.delete_file_chunks("codebase", rel_path)

    if not files_to_index:
        logger.info(
            "No files to re-embed. %d orphaned files cleaned up.", len(files_to_delete)
        )
        return

    # ----- 5. Chunk and embed new/modified files -----
    all_chunks: list[tuple[str, str, int, str, str]] = []
    for rel_path in files_to_index:
        file_hash, content = local_hashes[rel_path]
        parts = _chunk_text(content)
        for idx, part in enumerate(parts):
            all_chunks.append((rel_path, rel_path, idx, part, file_hash))

    logger.info("Total chunks to embed: %d", len(all_chunks))

    oai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    total_upserted = 0

    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c[3] for c in batch]
        logger.info(
            "Embedding batch %d–%d of %d",
            batch_start + 1,
            batch_start + len(batch),
            len(all_chunks),
        )
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
        await repo.upsert_chunks(records)
        total_upserted += len(records)
        logger.info("Upserted %d chunks (total: %d)", len(records), total_upserted)

    logger.info(
        "Incremental indexing complete. %d chunks stored, %d orphaned files removed.",
        total_upserted,
        len(files_to_delete),
    )


if __name__ == "__main__":
    asyncio.run(main())

