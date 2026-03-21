"""Minimal routes for the backend scaffold."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import ValidationError

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
    background_tasks: BackgroundTasks,
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

    background_tasks.add_task(container.intent.process_app_mention, envelope)
    return {"ok": True}
