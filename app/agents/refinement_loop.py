"""Phase 4 — autonomous critique-driven refinement loop over research drafts."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus

logger = logging.getLogger(__name__)

_CRITIQUE_SYSTEM_INSTRUCTION = """You are an exacting **research critique model** reviewing a draft\
 answer produced by another agent.

Goals:
1. Decide whether the draft **adequately resolves** the user's question with proportionate,\
 source-grounded depth.
2. List **specific gaps**: missing facets, unstated assumptions, weak evidence,\
 contradictions, or missing numeric/recency precision.
3. Propose **actionable follow-up sub-queries** a researcher armed with corpus + MCP +\
 web search could run next. Each query must be self-contained.

Output **only** compact JSON (**no prose outside JSON**) matching exactly:
{
  "satisfied": <boolean>,
  "gaps_identified": [<string>, ...],
  "follow_up_queries": [<string>, ...],
  "reviewer_notes": <string briefly summarising disposition>
}

`satisfied` MUST be **true** only when no material gaps remain credible; otherwise **false** with\
 substantive `follow_up_queries`. Prefer 1 to 4 follow-ups unless the gap is genuinely narrow.
"""

_CRITIQUE_USER_TEMPLATE = """ORIGINAL QUESTION:
{question}

CURRENT DRAFT ANSWER:
{draft}

Return only the critique JSON schema described in your instructions."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a single JSON object from model output (including ```json fenced blocks```)."""
    cleaned = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{(?:[^`]|`(?!``))*})\s*```", cleaned, re.DOTALL)
    blob = fenced_match.group(1).strip() if fenced_match else cleaned
    lb = blob.find("{")
    rb = blob.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        msg = "No JSON object found in critique response"
        raise ValueError(msg)
    return json.loads(blob[lb : rb + 1])


class CritiqueDecision(BaseModel):
    """Structured reviewer output."""

    satisfied: bool = Field(description="Reviewer believes draft is materially complete.")
    gaps_identified: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    reviewer_notes: str = ""


@dataclass(frozen=True)
class RefinementCycleRecord:
    """One critique evaluation after a draft."""

    critique_index: int
    critique: CritiqueDecision
    raw_critique_response: str


@dataclass(frozen=True)
class RefinementResearchRecord:
    """One research invocation inside the refinement loop."""

    round_index: int
    prompt: str
    answer: str


@dataclass(frozen=True)
class RefinementLoopResult:
    """Final artefacts from Phase 4."""

    original_question: str
    research_history: tuple[RefinementResearchRecord, ...]
    critiques: tuple[RefinementCycleRecord, ...]
    final_answer: str
    terminated_because_satisfied: bool


def build_follow_up_research_prompt(
    original_question: str,
    prior_answer: str,
    decision: CritiqueDecision,
) -> str:
    """Turn critique follow-ups into a single Phase-3-ready user prompt."""
    gap_block = "\n".join(f"- {g}" for g in decision.gaps_identified) or "- _(none enumerated)_"
    query_block = "\n".join(f"- {q}" for q in decision.follow_up_queries) or "- _(none)_"

    return (
        f"ORIGINAL USER QUESTION (keep as north star):\n{original_question}\n\n"
        "PRIOR AGENT ANSWER (refine/improve — do not ignore unless superseded):\n"
        f"{prior_answer}\n\n"
        f"ISSUES / GAPS TO ADDRESS:\n{gap_block}\n\n"
        f"ACTIONABLE SUB-QUESTIONS TO INVESTIGATE NOW:\n{query_block}\n\n"
        "Produce a tighter, evidence-grounded answer that closes these gaps (tools as needed)."
    )


async def _generate_critique_decision(
    settings: Settings,
    *,
    original_question: str,
    draft_answer: str,
    critique_model: str | None,
    raw_critique_snippet_max: int,
) -> tuple[CritiqueDecision, str]:
    configure_google_genai_api_key_environment(settings)
    model_id = critique_model or settings.gemini_model
    usr = _CRITIQUE_USER_TEMPLATE.format(
        question=original_question.strip(),
        draft=draft_answer.strip() or "_(empty)_",
    )
    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=usr,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=[_CRITIQUE_SYSTEM_INSTRUCTION],
        ),
    )
    raw = (response.text or "").strip()
    clipped = raw[:raw_critique_snippet_max] if raw else ""
    if not raw:
        logger.warning("Empty critique response; treating as satisfied with no gaps")
        return CritiqueDecision(satisfied=True, reviewer_notes="empty_critique_response"), clipped

    try:
        blob = extract_json_object(raw)
        decision = CritiqueDecision.model_validate(blob)
    except (ValidationError, ValueError):
        logger.exception("Failed to parse critique JSON; terminating refinement early")
        return (
            CritiqueDecision(
                satisfied=True,
                reviewer_notes="critique_parse_failed",
                gaps_identified=["Could not parse automated critique payload."],
            ),
            clipped,
        )

    logger.debug(
        "Critique satisfied=%s follow_ups=%d",
        decision.satisfied,
        len(decision.follow_up_queries),
    )
    return decision, clipped


async def default_phase3_runner(
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    *,
    user_id: str,
    session_id: str,
    round_marker: str,
) -> tuple[str, list[Any]]:
    """Default engine: Phase 3 MCP hybrid app run under a deterministic session slug."""
    # Local import avoids circular imports with ``session_runner`` (which shells ADK phases).
    from google.adk.apps.app import App
    from google.adk.events.event import Event
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types

    from app.agents.session_runner import concatenate_agent_text

    configure_google_genai_api_key_environment(settings)
    agent = create_phase3_mcp_agent(settings, corpus)
    app = App(name="phase4_refinement_leaf", root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    trimmed = question.strip()
    sess = f"{session_id}-{round_marker}"

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=sess,
        new_message=types.Content(role="user", parts=[types.Part(text=trimmed)]),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


async def run_phase4_refinement_loop(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    max_critique_iterations: int = 2,
    user_id: str = "phase4-user",
    session_id: str = "phase4-refinement-session",
    research_runner: (
        Callable[[Settings, KnowledgeCorpus, str, str], Awaitable[tuple[str, list[Any]]]] | None
    ) = None,
    critique_model: str | None = None,
    raw_critique_snippet_max: int = 8000,
) -> RefinementLoopResult:
    """Research then critique repeatedly until satisfaction or budget exhaustion.

    ``max_critique_iterations`` counts **post-draft refinement cycles**.
    Each cycle runs one JSON critique followed by — when still unsatisfied — one fresh research\
    Phase-3 pass. ``0`` disables critique entirely (initial draft only).

    Inject ``research_runner(settings, corpus, prompt, slug)`` to substitute the Phase-3 executor\
    inside tests or alternative planners.
    """

    if max_critique_iterations < 0:
        msg = "max_critique_iterations must be >= 0"
        raise ValueError(msg)

    trimmed_q = question.strip()
    research_history: list[RefinementResearchRecord] = []
    critiques: list[RefinementCycleRecord] = []

    async def invoke_research(prompt: str, slug: str) -> tuple[str, list[Any]]:
        if research_runner is not None:
            return await research_runner(settings, corpus, prompt, slug)
        return await default_phase3_runner(
            settings,
            corpus,
            prompt,
            user_id=user_id,
            session_id=session_id,
            round_marker=slug,
        )

    answer, _ev = await invoke_research(trimmed_q, "r0")
    research_history.append(
        RefinementResearchRecord(round_index=0, prompt=trimmed_q, answer=answer)
    )

    if max_critique_iterations == 0:
        return RefinementLoopResult(
            original_question=trimmed_q,
            research_history=tuple(research_history),
            critiques=tuple(critiques),
            final_answer=answer,
            terminated_because_satisfied=True,
        )

    for refinement_idx in range(max_critique_iterations):
        decision, raw_snippet = await _generate_critique_decision(
            settings,
            original_question=trimmed_q,
            draft_answer=answer,
            critique_model=critique_model,
            raw_critique_snippet_max=raw_critique_snippet_max,
        )

        critiques.append(
            RefinementCycleRecord(
                critique_index=refinement_idx,
                critique=decision,
                raw_critique_response=raw_snippet,
            )
        )

        if decision.satisfied:
            return RefinementLoopResult(
                original_question=trimmed_q,
                research_history=tuple(research_history),
                critiques=tuple(critiques),
                final_answer=answer,
                terminated_because_satisfied=True,
            )

        follow_prompt = build_follow_up_research_prompt(trimmed_q, answer, decision)
        slug = f"r{len(research_history)}"
        answer, _ev2 = await invoke_research(follow_prompt, slug)
        research_history.append(
            RefinementResearchRecord(
                round_index=len(research_history),
                prompt=follow_prompt,
                answer=answer,
            )
        )

    last_satisfied = bool(critiques) and critiques[-1].critique.satisfied

    if not last_satisfied:
        logger.info(
            "Stopping refinement — still unsatisfied after %d refinement cycle(s).",
            max_critique_iterations,
        )

    return RefinementLoopResult(
        original_question=trimmed_q,
        research_history=tuple(research_history),
        critiques=tuple(critiques),
        final_answer=answer,
        terminated_because_satisfied=last_satisfied,
    )


def run_phase4_refinement_loop_sync(**kwargs: Any) -> RefinementLoopResult:
    """Blocking helper for notebooks / Streamlit."""
    return asyncio.run(run_phase4_refinement_loop(**kwargs))
