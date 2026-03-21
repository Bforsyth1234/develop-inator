#!/usr/bin/env python3
"""Index a local repository into Supabase documentation_chunks for RAG retrieval.

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

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
BATCH_SIZE = 20  # embeddings per API call

# File extensions to index
INDEXABLE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json", ".css",
    ".html", ".yaml", ".yml", ".toml", ".sql", ".sh", ".env.example",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", ".next", "__pycache__", ".venv", "venv",
    "dist", "build", ".turbo", ".cache", "coverage",
}


def should_index(path: Path, repo_root: Path) -> bool:
    """Decide whether a file should be indexed."""
    rel = path.relative_to(repo_root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.suffix not in INDEXABLE_EXTENSIONS:
        return False
    # Skip very large files (>100KB)
    if path.stat().st_size > 100_000:
        return False
    return True


def collect_files(repo_root: Path) -> list[Path]:
    """Walk the repo and collect indexable files."""
    files = sorted(p for p in repo_root.rglob("*") if p.is_file() and should_index(p, repo_root))
    logger.info("Found %d indexable files in %s", len(files), repo_root)
    return files


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def embed_batch(client: openai.AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


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

    files = collect_files(repo_root)
    if not files:
        sys.exit("No indexable files found")

    oai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    transport = UrllibSupabaseTransport(base_url=SUPABASE_URL, service_role_key=SUPABASE_KEY)
    repo = DocumentationChunkRepository(transport)

    # Build all chunks first
    all_chunks: list[tuple[str, str, int, str]] = []  # (rel_path, source_id, chunk_idx, text)
    for file_path in files:
        rel = str(file_path.relative_to(repo_root))
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.warning("Skipping unreadable file: %s", rel)
            continue
        if not content.strip():
            continue
        parts = chunk_text(content)
        for idx, part in enumerate(parts):
            all_chunks.append((rel, rel, idx, part))

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # Process in batches
    total_upserted = 0
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c[3] for c in batch]
        logger.info("Embedding batch %d–%d of %d", batch_start + 1, batch_start + len(batch), len(all_chunks))
        vectors = await embed_batch(oai, texts)

        records = []
        for (rel_path, source_id, chunk_idx, text), vector in zip(batch, vectors):
            checksum = hashlib.sha256(text.encode()).hexdigest()[:16]
            records.append(
                DocumentationChunkRecord(
                    source_type="codebase",
                    source_id=source_id,
                    chunk_index=chunk_idx,
                    title=rel_path.split("/")[-1],
                    path=rel_path,
                    content=text,
                    embedding=tuple(vector),
                    metadata={"repo": str(repo_root.name)},
                    embedding_metadata=EmbeddingMetadata(
                        model=EMBEDDING_MODEL,
                        dimensions=len(vector),
                        source_checksum=checksum,
                    ),
                )
            )
        await repo.upsert_chunks(records)
        total_upserted += len(records)
        logger.info("Upserted %d chunks (total: %d)", len(records), total_upserted)

    logger.info("Indexing complete. %d chunks stored.", total_upserted)


if __name__ == "__main__":
    asyncio.run(main())

