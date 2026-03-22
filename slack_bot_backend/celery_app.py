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
    lock = redis_client.lock(lock_key, timeout=600, blocking_timeout=600)

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


@celery_app.task(name="slack_bot_backend.process_spec_approval", bind=True, max_retries=2)
def process_spec_approval_task(self, *, execution_id: str) -> None:
    """Execute an approved spec via ActionWorkflow._run_aider."""
    try:
        container = _get_container()
        _run_async(_execute_approved_spec(container, execution_id))
    except Exception as exc:
        logger.exception("process_spec_approval_task failed")
        raise self.retry(exc=exc, countdown=60) from exc


async def _execute_approved_spec(container, execution_id: str) -> None:
    """Async helper that mirrors the inline ``_execute_approved`` closure in routes."""
    from slack_bot_backend.models.action import ActionRequest as _AR

    execution = await container.supabase.get_action_execution(execution_id)
    if execution is None:
        logger.warning("Spec approval for unknown execution %s", execution_id)
        return

    req = _AR(
        channel=execution.channel,
        thread_ts=execution.thread_ts,
        request=execution.original_request,
        user_id=execution.user_id,
    )

    existing_branch: str | None = None
    try:
        mapping = await container.supabase.get_pr_mapping_by_thread(
            channel_id=execution.channel,
            thread_ts=execution.thread_ts,
        )
        if mapping is not None:
            existing_branch = mapping.branch_name
    except Exception:
        logger.warning("Could not look up PR mapping for thread", exc_info=True)

    selected_model = execution.model or container.action.model_tier_map.get("complex", "")
    try:
        aider_result = await container.action._run_aider(
            req,
            optimized_prompt=execution.generated_spec,
            model=selected_model,
            existing_branch=existing_branch,
        )
        if aider_result.returncode != 0:
            if aider_result.test_attempts > 0:
                await container.action._handle_test_failure(req, aider_result)
            else:
                await container.action._handle_failure(req, aider_result)
        else:
            await container.action._handle_success(req, aider_result, model=selected_model)
    except Exception:
        logger.exception("Approved spec execution failed")
        try:
            await container.slack.post_message(
                execution.channel,
                ":x: I hit an error executing the approved spec. Please check the logs.",
                thread_ts=execution.thread_ts,
            )
        except Exception:
            logger.warning("Failed to post error to Slack", exc_info=True)

