"""Construct the public `AgentCard` for the News Agent A2A endpoint."""

from __future__ import annotations

from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol


def build_news_kb_agent_card(*, public_base_url: str) -> AgentCard:
    """Return the News Agent card using REST (HTTP+JSON) at ``public_base_url``."""
    base = public_base_url.rstrip("/")
    iface = AgentInterface(
        url=base,
        protocol_binding=TransportProtocol.HTTP_JSON.value,
        protocol_version=PROTOCOL_VERSION_1_0,
    )
    skill = AgentSkill(
        id="latest_news",
        name="Latest topical news briefing",
        description=(
            "Search this server's ingested knowledge base (optional) and augment with "
            "hosted Google Search to brief on recent developments for a topic."
        ),
        tags=["news", "journalism", "knowledge-base"],
        examples=["What changed this week in EU AI regulation headlines?"],
    )
    caps = AgentCapabilities(
        streaming=False,
        push_notifications=False,
        extended_agent_card=False,
    )
    return AgentCard(
        name="news-knowledge-agent",
        description=(
            "Remote specialist that answers **latest news** questions using private "
            "uploads on this host plus grounded web search."
        ),
        supported_interfaces=[iface],
        capabilities=caps,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[skill],
    )
