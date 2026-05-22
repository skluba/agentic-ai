"""ADK-backed agents surfaced for product surfaces."""

from app.agents.core_rag import create_core_rag_agent
from app.agents.external_knowledge import create_external_knowledge_agent
from app.agents.news_focused import create_news_kb_agent
from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.agents.phase5_collaborative import create_phase5_collaborative_agent
from app.agents.refinement_loop import (
    RefinementLoopResult,
    run_phase4_refinement_loop,
    run_phase4_refinement_loop_sync,
)
from app.agents.session_runner import (
    run_core_rag_turn,
    run_core_rag_turn_sync,
    run_phase2_external_turn,
    run_phase2_external_turn_sync,
    run_phase3_mcp_turn,
    run_phase3_mcp_turn_sync,
    run_phase5_collaborative_turn,
    run_phase5_collaborative_turn_sync,
)

__all__ = [
    "RefinementLoopResult",
    "create_core_rag_agent",
    "create_external_knowledge_agent",
    "create_news_kb_agent",
    "create_phase3_mcp_agent",
    "create_phase5_collaborative_agent",
    "run_core_rag_turn",
    "run_core_rag_turn_sync",
    "run_phase2_external_turn",
    "run_phase2_external_turn_sync",
    "run_phase3_mcp_turn",
    "run_phase3_mcp_turn_sync",
    "run_phase4_refinement_loop",
    "run_phase4_refinement_loop_sync",
    "run_phase5_collaborative_turn",
    "run_phase5_collaborative_turn_sync",
]
