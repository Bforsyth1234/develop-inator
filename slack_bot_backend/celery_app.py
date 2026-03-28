"""Celery application and task definitions for durable background execution."""

from __future__ import annotations

import asyncio
import logging

import redis as _redis_lib
from celery import Celery

from slack_bot_backend.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
celery_app = Celery(
    "slack_bot_backend",
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def _get_container():
    """Build (or reuse) the ServiceContainer inside the Celery worker process."""
    from slack_bot_backend.dependencies import build_service_container

    # Cache the container on the module so it's built once per worker process.
    global _worker_container  # noqa: PLW0603
    try:
        return _worker_container
    except NameError:
        pass
    _worker_container = build_service_container(get_settings())
    return _worker_container


def _get_redis_client() -> _redis_lib.Redis:
    """Return a shared Redis client for distributed locking."""
    global _redis_client  # noqa: PLW0603
    try:
        return _redis_client
    except NameError:
        pass
    _redis_client = _redis_lib.Redis.from_url(settings.redis_url)
    return _redis_client


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop for this worker process.

    Unlike ``asyncio.run()`` (which creates *and closes* a new loop each time),
    this keeps a single loop alive so that cached ``httpx.AsyncClient`` instances
    (e.g. in ``GitHubGitService``) don't break with "Event loop is closed".
    """
    global _worker_loop  # noqa: PLW0603
    try:
        if not _worker_loop.is_closed():
            return _worker_loop
    except NameError:
        pass
    _worker_loop = asyncio.new_event_loop()
    return _worker_loop


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    loop = _get_worker_loop()
    return loop.run_until_complete(coro)


@celery_app.task(name="slack_bot_backend.process_slack_mention", bind=True, max_retries=2)
def process_slack_mention_task(self, envelope_json: str) -> None:
    """Process a Slack app_mention event via IntentWorkflow.

    Parameters
    ----------
    envelope_json:
        The raw JSON string of the :class:`SlackEventEnvelope`.
    """
    from slack_bot_backend.models.slack import SlackEventEnvelope

    try:
        envelope = SlackEventEnvelope.model_validate_json(envelope_json)
        container = _get_container()
        _run_async(container.intent.process_app_mention(envelope))
    except Exception as exc:
        logger.exception("process_slack_mention_task failed")
        raise self.retry(exc=exc, countdown=30) from exc


@celery_app.task(name="slack_bot_backend.process_github_webhook", bind=True, max_retries=2)
def process_github_webhook_task(self) -> None:
    """Trigger CodebaseIndexer.reindex from a GitHub push webhook."""
    try:
        container = _get_container()
        if container.indexer is None:
            logger.warning("process_github_webhook_task: indexer is not configured")
            return
        _run_async(container.indexer.reindex())
    except Exception as exc:
        logger.exception("process_github_webhook_task failed")
        raise self.retry(exc=exc, countdown=30) from exc


@celery_app.task(name="slack_bot_backend.process_pr_comment", bind=True, max_retries=2)
def process_pr_comment_task(
    self,
    *,
    pr_url: str,
    comment_body: str,
    sender: str,
    comment_node_id: str | None = None,
) -> None:
    """Handle a GitHub PR comment via ActionWorkflow.handle_pr_comment.

    A Redis distributed lock keyed on *pr_url* ensures only one comment is
    processed at a time per PR, even across multiple Celery worker processes.
    Without this, concurrent Aider runs on the same branch race each other
    and the losers fail with ``push rejected (stale info)``.
    """
    # Distributed lock: serialize all comment tasks for the same PR.
    # timeout  = max time the lock is held (auto-releases if worker dies).
    # blocking_timeout = max time a queued task waits to acquire the lock.
    redis_client = _get_redis_client()
    lock_key = f"pr-comment-lock:{pr_url}"
    lock = redis_client.lock(lock_key, timeout=1800, blocking_timeout=1800)

    try:
        acquired = lock.acquire(blocking=True)
        if not acquired:
            logger.warning("Could not acquire lock for %s; retrying task", pr_url)
            raise self.retry(countdown=15)

        container = _get_container()
        _run_async(
            container.action.handle_pr_comment(
                pr_url=pr_url,
                comment_body=comment_body,
                sender=sender,
                comment_node_id=comment_node_id,
            )
        )
    except self.MaxRetriesExceededError:
        logger.error("Max retries exceeded for PR comment on %s", pr_url)
    except Exception as exc:
        logger.exception("process_pr_comment_task failed")
        raise self.retry(exc=exc, countdown=30) from exc
    finally:
        try:
            lock.release()
        except _redis_lib.exceptions.LockNotOwnedError:
            pass  # Lock already expired or was never acquired




