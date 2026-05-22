"""A2A client helpers for invoking the standalone News Agent (HTTP+JSON)."""

from __future__ import annotations

import logging

import httpx
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.helpers.proto_helpers import get_message_text, new_text_message
from a2a.types.a2a_pb2 import Role, SendMessageConfiguration, SendMessageRequest
from a2a.utils.constants import TransportProtocol

logger = logging.getLogger(__name__)


async def delegate_latest_news_via_a2a(
    *,
    base_url: str,
    topic: str,
    time_horizon_hours: int = 72,
    http_timeout_seconds: float = 120.0,
) -> str:
    """Send ``topic`` to the remote News Agent and return plaintext (or Markdown) answer."""
    trimmed = topic.strip()
    bounds = max(1, min(int(time_horizon_hours), 24 * 30))
    if not trimmed:
        return "**A2A news client:** empty topic."

    effective_base = base_url.strip().rstrip("/")
    user_prompt = (
        f"Latest news briefing (last ~{bounds} hours context when relevant).\nTopic:\n{trimmed}"
    )

    outer_timeout = httpx.Timeout(http_timeout_seconds)
    inner_client = httpx.AsyncClient(timeout=outer_timeout)
    cfg = ClientConfig(
        streaming=False,
        polling=False,
        httpx_client=inner_client,
        supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
        use_client_preference=False,
    )
    factory = ClientFactory(cfg)
    resolved_client = None
    try:
        resolved_client = await factory.create_from_url(effective_base)
        req = SendMessageRequest(
            tenant="",
            message=new_text_message(text=user_prompt, role=Role.ROLE_USER),
            configuration=SendMessageConfiguration(),
        )
        response_text_parts: list[str] = []
        async for sr in resolved_client.send_message(req):
            if sr.HasField("message"):
                txt = get_message_text(sr.message)
                if txt:
                    response_text_parts.append(txt.strip())
                    break

        merged = "\n".join(part for part in response_text_parts if part)
        return merged or (
            "**A2A news client:** Received an empty acknowledgement from News Agent endpoint."
        )
    except httpx.TimeoutException:
        logger.warning(
            "A2A news client timed out base_url=%s timeout=%ss",
            effective_base,
            http_timeout_seconds,
        )
        return (
            "**A2A news client:** Request timed out — ensure the News Agent is running "
            f"(`{effective_base}`) and widen timeout if needed."
        )
    finally:
        if resolved_client is not None:
            await resolved_client.close()
        await inner_client.aclose()
