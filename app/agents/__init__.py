"""ADK-backed agents surfaced for product surfaces."""

from app.agents.core_rag import create_core_rag_agent
from app.agents.external_knowledge import create_external_knowledge_agent
from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.agents.session_runner import (
    run_core_rag_turn,
    run_core_rag_turn_sync,
    run_phase2_external_turn,
    run_phase2_external_turn_sync,
    run_phase3_mcp_turn,
    run_phase3_mcp_turn_sync,
)

__all__ = [
    "create_core_rag_agent",
    "create_external_knowledge_agent",
    "create_phase3_mcp_agent",
    "run_core_rag_turn",
    "run_core_rag_turn_sync",
    "run_phase2_external_turn",
    "run_phase2_external_turn_sync",
    "run_phase3_mcp_turn",
    "run_phase3_mcp_turn_sync",
]
