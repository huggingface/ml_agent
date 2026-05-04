"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

import asyncio
import json
import logging
from typing import Any

from dependencies import (
    INTERNAL_HF_TOKEN_KEY,
    get_current_user,
    require_huggingface_org_member,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from litellm import acompletion
from models import (
    ApprovalRequest,
    HealthResponse,
    LLMHealthResponse,
    SessionInfo,
    SessionNotificationsRequest,
    SessionResponse,
    SessionYoloRequest,
    SubmitRequest,
    TruncateRequest,
)
from session_manager import (
    MAX_SESSIONS,
    AgentSession,
    SessionCapacityError,
    session_manager,
)

import user_quotas

from agent.core.hf_access import get_jobs_access
from agent.core.hf_tokens import resolve_hf_request_token, resolve_hf_router_token
from agent.core.llm_params import _resolve_llm_params

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])
_background_teardown_tasks: set[asyncio.Task] = set()

DEFAULT_CLAUDE_MODEL_ID = "bedrock/us.anthropic.claude-opus-4-6-v1"
DEFAULT_FREE_MODEL_ID = "moonshotai/Kimi-K2.6"
GATED_MODEL_IDS = {
    DEFAULT_CLAUDE_MODEL_ID,
    "openai/gpt-5.5",
}


def _claude_picker_model_id() -> str:
    """Return the model ID used by the Claude option in the UI.

    The frontend config sets ``session_manager.config.model_name`` from
    ``ML_INTERN_CLAUDE_MODEL_ID`` when that env var is present, otherwise it
    falls back to the production Bedrock Claude model. This function only
    exposes that resolved config value for the Claude picker; non-Claude models
    are listed separately in the model switcher.
    """
    return session_manager.config.model_name


def _available_models() -> list[dict[str, Any]]:
    models = [
        {
            "id": "moonshotai/Kimi-K2.6",
            "label": "Kimi K2.6",
            "provider": "huggingface",
            "tier": "free",
            "recommended": True,
        },
        {
            "id": _claude_picker_model_id(),
            "label": "Claude Opus 4.6",
            "provider": "anthropic",
            "tier": "pro",
            "recommended": True,
        },
        {
            "id": "openai/gpt-5.5",
            "label": "GPT-5.5",
            "provider": "openai",
            "tier": "pro",
        },
        {
            "id": "MiniMaxAI/MiniMax-M2.7",
            "label": "MiniMax M2.7",
            "provider": "huggingface",
            "tier": "free",
        },
        {
            "id": "zai/glm-5.1",
            "label": "GLM 5.1",
            "provider": "zai",
            "tier": "free",
        },
        {
            "id": "deepseek-ai/DeepSeek-V4-Pro:deepinfra",
            "label": "DeepSeek V4 Pro",
            "provider": "huggingface",
            "tier": "free",
        },
    ]
    return models


AVAILABLE_MODELS = _available_models()


def _is_gated_model(model_id: str) -> bool:
    return model_id in GATED_MODEL_IDS


def _premium_model_restricted_error() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "error": "premium_model_restricted",
            "message": (
                "Premium models are gated to HF staff. Pick a free model — "
                "Kimi K2.6, MiniMax M2.7, GLM 5.1, or DeepSeek V4 Pro — "
                "instead."
            ),
        },
    )


async def _require_hf_for_gated_model(request: Request, model_id: str) -> None:
    """403 if a non-``huggingface``-org user tries to select a gated model.

    Gated models are deployed paid endpoints backed by service-owned
    credentials. The gate only fires for deployed paid models so non-HF users
    can still freely switch between the free models.
    """
    if not _is_gated_model(model_id):
        return
    if not await require_huggingface_org_member(request):
        raise _premium_model_restricted_error()


async def _model_override_for_new_session(
    request: Request,
    requested_model: str | None,
) -> str | None:
    """Return the model override to use when creating a new session.

    Explicit gated-model requests keep the hard membership gate. Implicit
    default sessions are more forgiving: when the configured default is gated
    and the user lacks access, start them on the first free model instead of
    blocking session creation.
    """
    resolved_model = requested_model or session_manager.config.model_name
    if not _is_gated_model(resolved_model):
        return requested_model
    if await require_huggingface_org_member(request):
        return requested_model
    if requested_model:
        raise _premium_model_restricted_error()

    logger.info(
        "Default gated model %s is unavailable to this user; "
        "creating session with free fallback %s",
        resolved_model,
        DEFAULT_FREE_MODEL_ID,
    )
    return DEFAULT_FREE_MODEL_ID


async def _enforce_gated_model_quota(
    user: dict[str, Any],
    agent_session: AgentSession,
) -> None:
    """Charge the user's daily gated-model quota on first use in a session.

    Runs at *message-submit* time, not session-create time — so spinning up a
    gated-model session to look around doesn't burn quota. The
    ``claude_counted`` flag on ``AgentSession`` guards against re-counting the
    same session; the stored field name is kept for persistence compatibility.

    No-ops when the session's current model isn't gated, or when this
    session has already been charged. Raises 429 when the user has hit
    their daily cap.
    """
    if agent_session.claude_counted:
        return
    model_name = agent_session.session.config.model_name
    if not _is_gated_model(model_name):
        return
    user_id = user["user_id"]
    cap = user_quotas.daily_cap_for(user.get("plan"))
    new_count = await user_quotas.try_increment_claude(user_id, cap)
    if new_count is None:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "premium_model_daily_cap",
                "plan": user.get("plan", "free"),
                "cap": cap,
                "message": (
                    "Daily premium model limit reached. Upgrade to HF Pro for "
                    f"{user_quotas.CLAUDE_PRO_DAILY}/day or use a free model."
                ),
            },
        )
    agent_session.claude_counted = True
    await session_manager.persist_session_snapshot(agent_session)


def _user_hf_token(user: dict[str, Any] | None) -> str | None:
    if not isinstance(user, dict):
        return None
    return user.get(INTERNAL_HF_TOKEN_KEY)


async def _check_session_access(
    session_id: str,
    user: dict[str, Any],
    request: Request | None = None,
    preload_sandbox: bool = True,
) -> AgentSession:
    """Verify and lazily load the user's session. Raises 403 or 404."""
    hf_token = (
        resolve_hf_request_token(request)
        if request is not None
        else _user_hf_token(user)
    )
    agent_session = await session_manager.ensure_session_loaded(
        session_id,
        user["user_id"],
        hf_token=hf_token,
        hf_username=user.get("username"),
        preload_sandbox=preload_sandbox,
    )
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if user["user_id"] != "dev" and agent_session.user_id not in {
        user["user_id"],
        "dev",
    }:
        raise HTTPException(status_code=403, detail="Access denied to this session")
    return agent_session


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
    )


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call.  Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = session_manager.config.model_name
    try:
        llm_params = _resolve_llm_params(model, reasoning_effort="high")
        await acompletion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=10,
            **llm_params,
        )
        return LLMHealthResponse(status="ok", model=model)
    except Exception as e:
        err_str = str(e).lower()
        error_type = "unknown"

        if (
            "401" in err_str
            or "auth" in err_str
            or "invalid" in err_str
            or "api key" in err_str
        ):
            error_type = "auth"
        elif (
            "402" in err_str
            or "credit" in err_str
            or "quota" in err_str
            or "insufficient" in err_str
            or "billing" in err_str
        ):
            error_type = "credits"
        elif "429" in err_str or "rate" in err_str:
            error_type = "rate_limit"
        elif "timeout" in err_str or "connect" in err_str or "network" in err_str:
            error_type = "network"

        logger.warning(f"LLM health check failed ({error_type}): {e}")
        return LLMHealthResponse(
            status="error",
            model=model,
            error=str(e)[:500],
            error_type=error_type,
        )


@router.get("/config/model")
async def get_model() -> dict:
    """Get current model and available models. No auth required."""
    return {
        "current": session_manager.config.model_name,
        "available": AVAILABLE_MODELS,
    }


_TITLE_STRIP_CHARS = str.maketrans("", "", "`*_~#[]()")


@router.post("/title")
async def generate_title(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Generate a short title for a chat session based on the first user message.

    Always uses gpt-oss-120b via Cerebras on the HF router. The tab headline
    renders as plain text, so the model is told to avoid markdown and any
    stray formatting characters are stripped before returning. gpt-oss is a
    reasoning model — reasoning_effort=low keeps the reasoning budget small
    so the 60-token output budget isn't consumed before the title is written.
    """
    api_key = resolve_hf_router_token(_user_hf_token(user))
    try:
        response = await acompletion(
            # Double openai/ prefix: LiteLLM strips the first as its provider
            # prefix, leaving the HF model id on the wire for the router.
            model="openai/openai/gpt-oss-120b:cerebras",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words) for a chat conversation "
                        "that starts with the following user message. "
                        "Reply with ONLY the title in plain text. "
                        "Do NOT use markdown, backticks, asterisks, quotes, brackets, or any "
                        "formatting characters. No punctuation at the end."
                    ),
                },
                {"role": "user", "content": request.text[:500]},
            ],
            max_tokens=60,
            temperature=0.3,
            timeout=10,
            reasoning_effort="low",
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        title = title.translate(_TITLE_STRIP_CHARS).strip()
        if len(title) > 50:
            title = title[:50].rstrip() + "…"
        try:
            await _check_session_access(request.session_id, user)
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping title persistence for missing session %s", request.session_id
            )
        return {"title": title}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        fallback = request.text.strip()
        title = fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback
        try:
            await _check_session_access(request.session_id, user)
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping fallback title persistence for missing session %s",
                request.session_id,
            )
        return {"title": title}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new agent session bound to the authenticated user.

    The user's HF access token is extracted from the Authorization header
    and stored in the session so that tools (e.g. hf_jobs) can act on
    behalf of the user.

    Optional body ``{"model"?: <id>}`` selects the session's LLM; unknown
    ids are rejected (400). The gated-model quota runs at message-submit
    time, not here — spinning up a session to look around is free.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header, HttpOnly cookie, or env var)
    hf_token = resolve_hf_request_token(request)

    # Optional model override. Empty body falls back to the config default.
    model: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        model = body.get("model")

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Explicit premium selections remain gated. If the implicit configured
    # default is unavailable, start the session on a free model instead.
    model = await _model_override_for_new_session(request, model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            model=model,
            is_pro=user.get("plan") == "pro",
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model or session_manager.config.model_name,
    )


@router.post("/session/restore-summary", response_model=SessionResponse)
async def restore_session_summary(
    request: Request, body: dict, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new session seeded with a summary of the caller's prior
    conversation. The client sends its cached messages; we run the standard
    summarization prompt on them and drop the result into the new
    session's context as a user-role system note.

    Optional ``"model"`` in the body overrides the session's LLM. The
    gated-model quota runs at message-submit time, not here.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' array")

    hf_token = resolve_hf_request_token(request)

    model = body.get("model")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    model = await _model_override_for_new_session(request, model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            model=model,
            is_pro=user.get("plan") == "pro",
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        summarized = await session_manager.seed_from_summary(session_id, messages)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("seed_from_summary failed")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    logger.info(
        f"Seeded session {session_id} for {user.get('username', 'unknown')} "
        f"(summary of {summarized} messages)"
    )
    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model or session_manager.config.model_name,
    )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    await _check_session_access(session_id, user)
    info = session_manager.get_session_info(session_id)
    return SessionInfo(**info)


@router.post("/session/{session_id}/model")
async def set_session_model(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Switch the active model for a single session (tab-scoped).

    Takes effect on the next LLM call in that session — other sessions
    (including other browser tabs) are unaffected. Model switches don't
    charge quota — the gated-model quota only fires at message-submit time.

    Switching TO a gated deployed model requires HF org membership; free-model
    and local-dev direct provider switches are unrestricted.
    """
    agent_session = await _check_session_access(session_id, user, request)
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    await _require_hf_for_gated_model(request, model_id)
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    await session_manager.update_session_model(session_id, model_id)
    logger.info(
        f"Session {session_id} model → {model_id} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {"session_id": session_id, "model": model_id}


@router.post("/session/{session_id}/notifications")
async def set_session_notifications(
    session_id: str,
    body: SessionNotificationsRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Replace the session's auto-notification destinations."""
    agent_session = await _check_session_access(session_id, user)
    try:
        destinations = session_manager.set_notification_destinations(
            session_id, body.destinations
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await session_manager.persist_session_snapshot(agent_session)
    return {
        "session_id": session_id,
        "notification_destinations": destinations,
    }


@router.patch("/session/{session_id}/yolo")
async def set_session_yolo(
    session_id: str,
    body: SessionYoloRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Update the session-scoped auto-approval policy."""
    await _check_session_access(session_id, user)
    try:
        summary = await session_manager.update_session_auto_approval(
            session_id,
            enabled=body.enabled,
            cost_cap_usd=body.cost_cap_usd,
            cap_provided="cost_cap_usd" in body.model_fields_set,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, **summary}


@router.get("/user/quota")
async def get_user_quota(user: dict = Depends(get_current_user)) -> dict:
    """Return the user's plan tier and today's premium-model quota state."""
    plan = user.get("plan", "free")
    used = await user_quotas.get_claude_used_today(user["user_id"])
    cap = user_quotas.daily_cap_for(plan)
    remaining = max(0, cap - used)
    return {
        "plan": plan,
        "premium_used_today": used,
        "premium_daily_cap": cap,
        "premium_remaining": remaining,
    }


@router.get("/user/jobs-access")
async def get_jobs_access_info(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Return the namespaces the current token can run HF Jobs under.

    Credits are enforced by the HF API at job-creation time, not here —
    the response only describes which wallets the caller is allowed to
    pick from. Pro is irrelevant.
    """
    token = resolve_hf_request_token(request)

    access = await get_jobs_access(token or "")
    return {
        "eligible_namespaces": access.eligible_namespaces if access else [],
        "default_namespace": access.default_namespace if access else None,
        "billing_url": "https://huggingface.co/settings/billing",
    }


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = await session_manager.list_sessions(user_id=user["user_id"])
    return [SessionInfo(**s) for s in sessions]


@router.post("/session/{session_id}/sandbox/teardown")
async def teardown_session_sandbox(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Best-effort sandbox teardown that preserves durable chat history."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    task = asyncio.create_task(session_manager.teardown_sandbox(session_id))
    _background_teardown_tasks.add(task)
    task.add_done_callback(_background_teardown_tasks.discard)
    return {"status": "teardown_requested", "session_id": session_id}


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    agent_session = await _check_session_access(request.session_id, user)
    await _enforce_gated_model_quota(user, agent_session)
    success = await session_manager.submit_user_input(request.session_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    await _check_session_access(request.session_id, user)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
            "namespace": a.namespace,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/chat/{session_id}")
async def chat_sse(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """SSE endpoint: submit input or approval, then stream events until turn ends."""
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    # Parse body
    body = await request.json()

    # Subscribe BEFORE submitting so we never miss events — even if the
    # agent loop processes the submission before this coroutine continues.
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()

    # Submit the operation
    text = body.get("text")
    approvals = body.get("approvals")

    # Gate user-message sends against the daily gated-model quota. Approvals are
    # continuations of an in-progress turn — the session was already charged
    # on its first message, so we skip the gate there.
    if text is not None and not approvals:
        try:
            await _enforce_gated_model_quota(user, agent_session)
        except HTTPException:
            broadcaster.unsubscribe(sub_id)
            raise

    try:
        if approvals:
            formatted = [
                {
                    "tool_call_id": a["tool_call_id"],
                    "approved": a["approved"],
                    "feedback": a.get("feedback"),
                    "edited_script": a.get("edited_script"),
                    "namespace": a.get("namespace"),
                }
                for a in approvals
            ]
            success = await session_manager.submit_approval(session_id, formatted)
        elif text is not None:
            success = await session_manager.submit_user_input(session_id, text)
        else:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(
                status_code=400, detail="Must provide 'text' or 'approvals'"
            )

        if not success:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        broadcaster.unsubscribe(sub_id)
        raise
    except Exception:
        broadcaster.unsubscribe(sub_id)
        raise

    return _sse_response(broadcaster, event_queue, sub_id)


@router.post("/pro-click/{session_id}")
async def record_pro_click(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Record a click on a Pro upgrade CTA shown from inside a session."""
    agent_session = await _check_session_access(session_id, user)

    from agent.core import telemetry

    await telemetry.record_pro_cta_click(
        agent_session.session,
        source=str(body.get("source") or "unknown"),
        target=str(body.get("target") or "pro_pricing"),
    )
    if agent_session.session.config.save_sessions:
        agent_session.session.save_and_upload_detached(
            agent_session.session.config.session_dataset_repo
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Shared SSE helpers
# ---------------------------------------------------------------------------
_TERMINAL_EVENTS = {
    "turn_complete",
    "approval_required",
    "error",
    "interrupted",
    "shutdown",
}
_SSE_KEEPALIVE_SECONDS = 15


def _last_event_seq(request: Request) -> int:
    raw = (
        request.headers.get("last-event-id") or request.query_params.get("after") or "0"
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _format_sse(msg: dict[str, Any]) -> str:
    seq = msg.get("seq")
    body = {"event_type": msg.get("event_type"), "data": msg.get("data") or {}}
    if seq is not None:
        body["seq"] = seq
        return f"id: {seq}\ndata: {json.dumps(body)}\n\n"
    return f"data: {json.dumps(body)}\n\n"


def _event_doc_to_msg(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": doc.get("event_type"),
        "data": doc.get("data") or {},
        "seq": doc.get("seq"),
    }


def _sse_response(
    broadcaster,
    event_queue,
    sub_id,
    *,
    replay_events: list[dict[str, Any]] | None = None,
    after_seq: int = 0,
) -> StreamingResponse:
    """Build a StreamingResponse that drains *event_queue* as SSE,
    sending keepalive comments every 15 s to prevent proxy timeouts."""

    async def event_generator():
        try:
            for doc in replay_events or []:
                msg = _event_doc_to_msg(doc)
                seq = msg.get("seq")
                if isinstance(seq, int) and seq <= after_seq:
                    continue
                yield _format_sse(msg)
                if msg.get("event_type", "") in _TERMINAL_EVENTS:
                    return

            while True:
                try:
                    msg = await asyncio.wait_for(
                        event_queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    # SSE comment — ignored by parsers, keeps connection alive
                    yield ": keepalive\n\n"
                    continue
                event_type = msg.get("event_type", "")
                yield _format_sse(msg)
                if event_type in _TERMINAL_EVENTS:
                    break
        finally:
            broadcaster.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/{session_id}")
async def subscribe_events(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to events for a running session without submitting new input.

    Used by the frontend to re-attach after a connection drop (e.g. screen
    sleep).  Returns 404 if the session isn't active or isn't processing.
    """
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    after_seq = _last_event_seq(request)
    replay_events = await session_manager._store().load_events_after(
        session_id, after_seq
    )
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        replay_events=replay_events,
        after_seq=after_seq,
    )


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, user: dict = Depends(get_current_user)
) -> list[dict]:
    """Return the session's message history from memory."""
    agent_session = await _check_session_access(session_id, user)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return [
        msg.model_dump(mode="json")
        for msg in agent_session.session.context_manager.items
    ]


@router.post("/undo/{session_id}")
async def undo_session(session_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Undo the last turn in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/truncate/{session_id}")
async def truncate_session(
    session_id: str, body: TruncateRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Truncate conversation to before a specific user message."""
    await _check_session_access(session_id, user)
    success = await session_manager.truncate(session_id, body.user_message_index)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found, inactive, or message index out of range",
        )
    return {"status": "truncated", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}


@router.post("/feedback/{session_id}")
async def submit_feedback(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Attach a user feedback signal to a session's event log.

    Body: {rating: "up"|"down"|"outcome_success"|"outcome_fail",
           turn_index?: int, comment?: str, message_id?: str}
    Appended as a `feedback` event and saved with the session trajectory.
    """
    agent_session = await _check_session_access(session_id, user)

    rating = body.get("rating")
    if rating not in {"up", "down", "outcome_success", "outcome_fail"}:
        raise HTTPException(status_code=400, detail="invalid rating")

    from agent.core import telemetry

    await telemetry.record_feedback(
        agent_session.session,
        rating=rating,
        turn_index=body.get("turn_index"),
        message_id=body.get("message_id"),
        comment=body.get("comment"),
    )
    # Fire-and-forget save so feedback reaches the dataset even if the user
    # closes the tab right after clicking.
    if agent_session.session.config.save_sessions:
        agent_session.session.save_and_upload_detached(
            agent_session.session.config.session_dataset_repo
        )
    return {"status": "ok"}
