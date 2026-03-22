"""Minimal routes for the backend scaffold."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from slack_bot_backend.celery_app import (
    process_github_webhook_task,
    process_pr_comment_task,
    process_slack_mention_task,
    process_spec_approval_task,
)
from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, get_container
from slack_bot_backend.models import ActionRequest, ActionRouteResult, QuestionRequest, QuestionRouteResult
from slack_bot_backend.models.slack import SlackEventEnvelope

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_slack_signature(settings: Settings, request: Request, body: bytes) -> None:
    if not settings.slack_signing_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack signing secret is not configured",
        )

    timestamp = request.headers.get("x-slack-request-timestamp")
    signature = request.headers.get("x-slack-signature")
    if not timestamp or not signature:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Slack signature")

    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Slack timestamp") from exc

    if abs(time.time() - request_time) > settings.slack_request_tolerance_seconds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expired Slack request")

    signature_basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode("utf-8"),
        signature_basestring,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Slack signature")


@router.get("/healthz", tags=["health"])
async def healthcheck(container: ServiceContainer = Depends(get_container)) -> dict[str, object]:
    return {
        "status": "ok",
        "environment": container.settings.environment,
        "providers": {
            "slack": type(container.slack).__name__,
            "supabase": type(container.supabase).__name__,
            "llm": type(container.llm).__name__,
            "git": type(container.git).__name__,
        },
    }


@router.post("/question", response_model=QuestionRouteResult, tags=["question"])
async def handle_question(
    payload: QuestionRequest,
    container: ServiceContainer = Depends(get_container),
) -> QuestionRouteResult:
    return await container.question.run(payload)


@router.post("/action", response_model=ActionRouteResult, tags=["action"])
async def handle_action(
    payload: ActionRequest,
    container: ServiceContainer = Depends(get_container),
) -> ActionRouteResult:
    return await container.action.run(payload)


@router.post("/slack/events", tags=["slack"])
async def handle_slack_events(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> dict[str, object]:
    body = await request.body()
    _verify_slack_signature(container.settings, request, body)

    try:
        envelope = SlackEventEnvelope.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Slack payload") from exc

    if envelope.type == "url_verification":
        if not envelope.challenge:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Slack challenge",
            )
        return {"challenge": envelope.challenge}

    if request.headers.get("x-slack-retry-num"):
        logger.info(
            "Ignoring Slack retry request",
            extra={
                "retry_num": request.headers.get("x-slack-retry-num"),
                "retry_reason": request.headers.get("x-slack-retry-reason"),
            },
        )
        return {"ok": True}

    if envelope.type != "event_callback" or envelope.event is None:
        return {"ok": True}

    if envelope.event.type != "app_mention" or envelope.event.bot_id or envelope.event.subtype:
        return {"ok": True}

    process_slack_mention_task.delay(body.decode("utf-8"))
    return {"ok": True}


# ---------------------------------------------------------------------------
# GitHub webhook — triggers re-indexing on pushes to the default branch
# ---------------------------------------------------------------------------


def _verify_github_signature(secret: str, body: bytes, signature_header: str | None) -> None:
    """Verify the X-Hub-Signature-256 header from GitHub."""
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing GitHub signature")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid GitHub signature")


@router.post("/github/webhook", tags=["github"])
async def handle_github_webhook(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> dict[str, object]:
    body = await request.body()

    # Verify signature if a webhook secret is configured
    if container.settings.github_webhook_secret:
        _verify_github_signature(
            container.settings.github_webhook_secret,
            body,
            request.headers.get("x-hub-signature-256"),
        )

    event_type = request.headers.get("x-github-event")
    if event_type == "ping":
        return {"ok": True, "message": "pong"}

    try:
        payload = json.loads(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload"
        ) from exc

    # ── Handle PR comment events ──
    if event_type == "issue_comment":
        return await _handle_issue_comment(payload, container)

    if event_type == "pull_request_review_comment":
        return await _handle_review_comment(payload, container)

    if event_type == "pull_request_review":
        # Review-level events are typically summaries ("Review completed.
        # 3 suggestions posted.") that duplicate the inline comments we
        # already handle via pull_request_review_comment.  Skip them to
        # avoid duplicate / non-actionable Aider runs.
        return {"ok": True, "message": "pull_request_review events are handled via individual review comments"}

    if event_type != "push":
        return {"ok": True, "message": f"ignored event: {event_type}"}

    ref: str = payload.get("ref", "")
    default_branch = payload.get("repository", {}).get("default_branch", "main")

    # Only re-index on pushes to the default branch (e.g. refs/heads/main)
    if ref != f"refs/heads/{default_branch}":
        logger.info(
            "Ignoring push to non-default branch",
            extra={"ref": ref, "default_branch": default_branch},
        )
        return {"ok": True, "message": f"ignored ref: {ref}"}

    if container.indexer is None:
        logger.warning("GitHub webhook received but indexer is not configured")
        return {"ok": False, "message": "indexer not configured"}

    logger.info(
        "GitHub push to default branch detected — scheduling re-index",
        extra={"ref": ref, "default_branch": default_branch},
    )
    process_github_webhook_task.delay()
    return {"ok": True, "message": "re-index scheduled"}


# ---------------------------------------------------------------------------
# GitHub issue_comment handler (PR comment feedback loop)
# ---------------------------------------------------------------------------


async def _handle_issue_comment(
    payload: dict,
    container: ServiceContainer,
) -> dict[str, object]:
    """Dispatch a GitHub ``issue_comment`` event if it's on a PR we created."""
    # Only handle newly created comments
    if payload.get("action") != "created":
        return {"ok": True, "message": "ignored non-created comment action"}

    # Must be a PR (not a plain issue)
    issue = payload.get("issue", {})
    pr_data = issue.get("pull_request")
    if not pr_data:
        return {"ok": True, "message": "ignored comment on non-PR issue"}

    # Ignore comments made by the bot itself
    sender_login: str = payload.get("sender", {}).get("login", "")
    if sender_login == container.settings.github_bot_username:
        return {"ok": True, "message": "ignored bot's own comment"}

    comment_body: str = payload.get("comment", {}).get("body", "")
    pr_url: str = pr_data.get("html_url", "") or issue.get("html_url", "")

    if not comment_body or not pr_url:
        return {"ok": True, "message": "missing comment body or PR URL"}

    logger.info(
        "GitHub PR comment received — dispatching handler",
        extra={"pr_url": pr_url, "sender": sender_login},
    )

    process_pr_comment_task.apply_async(
        kwargs={
            "pr_url": pr_url,
            "comment_body": comment_body,
            "sender": sender_login,
        },
    )
    return {"ok": True, "message": "pr comment handler scheduled"}


# ---------------------------------------------------------------------------
# GitHub pull_request_review_comment handler (inline code review comments)
# ---------------------------------------------------------------------------


async def _handle_review_comment(
    payload: dict,
    container: ServiceContainer,
) -> dict[str, object]:
    """Dispatch a GitHub ``pull_request_review_comment`` event (inline code comment)."""
    if payload.get("action") != "created":
        return {"ok": True, "message": "ignored non-created review comment action"}

    sender_login: str = payload.get("sender", {}).get("login", "")
    if sender_login == container.settings.github_bot_username:
        return {"ok": True, "message": "ignored bot's own review comment"}

    comment = payload.get("comment", {})
    comment_body: str = comment.get("body", "")
    comment_node_id: str = comment.get("node_id", "")
    diff_hunk: str = comment.get("diff_hunk", "")
    path: str = comment.get("path", "")

    pr_url: str = payload.get("pull_request", {}).get("html_url", "")
    if not comment_body or not pr_url:
        return {"ok": True, "message": "missing comment body or PR URL"}

    # Include file context so the bot understands _where_ the comment applies
    enriched_body = comment_body
    if path:
        enriched_body = f"[{path}]\n{comment_body}"
    if diff_hunk:
        enriched_body = f"{enriched_body}\n\nRelevant diff:\n```\n{diff_hunk}\n```"

    logger.info(
        "GitHub PR inline review comment received",
        extra={"pr_url": pr_url, "sender": sender_login, "path": path},
    )

    process_pr_comment_task.apply_async(
        kwargs={
            "pr_url": pr_url,
            "comment_body": enriched_body,
            "sender": sender_login,
            "comment_node_id": comment_node_id or None,
        },
    )
    return {"ok": True, "message": "pr review comment handler scheduled"}


# ---------------------------------------------------------------------------
# Slack interactive components (Block Kit buttons)
# ---------------------------------------------------------------------------


async def _handle_spec_action(
    container: ServiceContainer,
    payload: dict,
) -> dict[str, object]:
    """Process approve_spec / reject_spec button clicks."""
    actions = payload.get("actions", [])
    if not actions:
        return {"ok": True}

    action = actions[0]
    action_id: str = action.get("action_id", "")
    execution_id: str = action.get("value", "")

    if not execution_id:
        return {"ok": True}

    execution = await container.supabase.get_action_execution(execution_id)
    if execution is None:
        logger.warning("Interaction for unknown execution %s", execution_id)
        return {"ok": True}

    if action_id == "approve_spec":
        await container.supabase.update_action_execution_status(execution_id, "approved")

        # Update the original Slack message to reflect approval
        message_container = payload.get("container", {})
        message_ts = message_container.get("message_ts", "")
        if message_ts:
            try:
                await container.slack.update_message(
                    execution.channel,
                    message_ts,
                    ":white_check_mark: Spec approved — executing…",
                )
            except Exception:
                logger.warning("Could not update approval message", exc_info=True)

        # Kick off Aider execution via Celery task
        process_spec_approval_task.apply_async(
            kwargs={"execution_id": execution_id},
        )
        return {"ok": True}

    if action_id == "reject_spec":
        await container.supabase.update_action_execution_status(execution_id, "rejected")
        message_container = payload.get("container", {})
        message_ts = message_container.get("message_ts", "")
        if message_ts:
            try:
                await container.slack.update_message(
                    execution.channel,
                    message_ts,
                    ":x: Spec rejected. Reply in the thread with feedback to generate a new one.",
                )
            except Exception:
                logger.warning("Could not update rejection message", exc_info=True)
        return {"ok": True}

    return {"ok": True}


@router.post("/slack/interactions", tags=["slack"])
async def handle_slack_interactions(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> dict[str, object]:
    """Handle Slack interactive component payloads (Block Kit buttons, etc.)."""
    body = await request.body()
    _verify_slack_signature(container.settings, request, body)

    # Slack sends interaction payloads as application/x-www-form-urlencoded
    # with a single `payload` field containing JSON.
    decoded = urllib.parse.parse_qs(body.decode("utf-8"))
    raw_payload = decoded.get("payload", [None])[0]
    if not raw_payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing payload")

    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc

    interaction_type = payload.get("type", "")
    if interaction_type == "block_actions":
        return await _handle_spec_action(container, payload)

    return {"ok": True}