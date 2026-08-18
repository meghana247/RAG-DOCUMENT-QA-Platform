import json
import logging
import uuid

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.core.dependencies import AppSettings, ChatServiceDep, CurrentUser, PaginationDep
from app.schemas.chat import (
    AnswerResponse,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    MessageCreate,
)
from app.schemas.common import Page

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(
    body: ChatSessionCreate, service: ChatServiceDep, user: CurrentUser
) -> ChatSessionResponse:
    chat_session = await service.create_session(user.id, body.document_id, body.title)
    return ChatSessionResponse.model_validate(chat_session)


@router.get("/sessions", response_model=Page[ChatSessionResponse])
async def list_sessions(
    service: ChatServiceDep, user: CurrentUser, page: PaginationDep
) -> Page[ChatSessionResponse]:
    sessions, total = await service.list_sessions(user.id, page.limit, page.offset)
    return Page[ChatSessionResponse](
        items=[ChatSessionResponse.model_validate(s) for s in sessions],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post("/sessions/{session_id}/messages", response_model=AnswerResponse)
async def send_message(
    session_id: uuid.UUID,
    body: MessageCreate,
    service: ChatServiceDep,
    settings: AppSettings,
    user: CurrentUser,
) -> AnswerResponse:
    message, answer = await service.send_message(session_id, user.id, body.content)
    return AnswerResponse(
        message=ChatMessageResponse.model_validate(message),
        grounded=answer.grounded,
        search_query=answer.search_query,
        timings=(
            answer.timings.as_dict()
            if settings.observability.expose_debug_timings
            else None
        ),
    )


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    session_id: uuid.UUID,
    body: MessageCreate,
    service: ChatServiceDep,
    user: CurrentUser,
) -> StreamingResponse:
    """Server-sent events: `sources`, then `token` deltas, then `done`.

    Sources are emitted before the first token so a client can render citations
    alongside the answer as it arrives rather than after it completes.
    """
    sources, stream, finalize = await service.stream_message(
        session_id, user.id, body.content
    )

    async def events():
        yield _sse("sources", {"sources": [s.to_citation() for s in sources]})

        collected: list[str] = []
        try:
            async for delta in stream:
                collected.append(delta)
                yield _sse("token", {"delta": delta})
        except Exception as exc:
            # The response has already begun, so the status code is committed;
            # the only way to report a mid-stream failure is as an event.
            logger.error("stream failed mid-response", exc_info=exc)
            yield _sse("error", {"message": "Generation failed before completing"})

        answer = "".join(collected)
        if answer:
            message = await finalize(answer)
            yield _sse("done", {"message_id": str(message.id)})
        else:
            yield _sse("done", {"message_id": None})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into one delivery.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}/messages", response_model=Page[ChatMessageResponse])
async def list_messages(
    session_id: uuid.UUID,
    service: ChatServiceDep,
    user: CurrentUser,
    page: PaginationDep,
) -> Page[ChatMessageResponse]:
    messages, total = await service.list_messages(
        session_id, user.id, page.limit, page.offset
    )
    return Page[ChatMessageResponse](
        items=[ChatMessageResponse.model_validate(m) for m in messages],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID, service: ChatServiceDep, user: CurrentUser
) -> None:
    await service.delete_session(session_id, user.id)
