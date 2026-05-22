"""Phase 4 refinement loop orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import app.agents.refinement_loop as refinement_loop
import pytest
from app.agents.refinement_loop import (
    CritiqueDecision,
    RefinementResearchRecord,
    build_follow_up_research_prompt,
    extract_json_object,
    run_phase4_refinement_loop,
)
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus


def test_extract_json_object_plain_and_fenced():
    blob = '{"satisfied":true,"gaps_identified":[],"follow_up_queries":[],"reviewer_notes":"ok"}'
    assert extract_json_object(blob)["satisfied"] is True
    fenced = f"```json\n{blob}\n```"
    assert extract_json_object(fenced)["satisfied"] is True


def test_build_follow_up_contains_gaps_queries():
    d = CritiqueDecision(
        satisfied=False,
        gaps_identified=["Missing risk section"],
        follow_up_queries=["List top three KPIs"],
    )
    merged = build_follow_up_research_prompt("What should we ship?", "draft", d)
    assert "risk" in merged.lower()
    assert "KPI" in merged


def test_extract_raises_on_missing_brackets():
    with pytest.raises(ValueError, match="No JSON"):
        extract_json_object("just prose without JSON")


async def fake_research(*_args, **_kwargs) -> tuple[str, list]:
    await asyncio.sleep(0)
    return "SYNTHESIS", []


def test_phase4_zero_skips_critique():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    settings = Settings(gcp_project_id="phase4-zero")

    async def body():
        return await run_phase4_refinement_loop(
            settings=settings,
            corpus=corpus,
            question="hello",
            max_critique_iterations=0,
            research_runner=fake_research,
        )

    seen = asyncio.run(body())
    assert seen.final_answer == "SYNTHESIS"
    assert seen.critiques == ()
    assert len(seen.research_history) == 1


def test_phase4_negative_max_iterations():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    settings = Settings(gcp_project_id="phase4-bad-max")

    async def body():
        await run_phase4_refinement_loop(
            settings=settings,
            corpus=corpus,
            question="Q",
            max_critique_iterations=-1,
            research_runner=fake_research,
        )

    with pytest.raises(ValueError, match=">="):
        asyncio.run(body())


def test_phase4_stops_when_critique_satisfied_after_refine():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    settings = Settings(gcp_project_id="phase4-exit")

    calls: list[str] = []

    async def tracking_research(_s, _c, prompt: str, slug: str) -> tuple[str, list]:
        await asyncio.sleep(0)
        calls.append(slug)
        return f"DONE-{slug}", []

    decisions = iter(
        [
            (CritiqueDecision(satisfied=False, follow_up_queries=["dig deeper"]), '{"a": 1}'),
            (CritiqueDecision(satisfied=True, reviewer_notes="done"), "{}"),
        ]
    )

    async def pretend_critique(*_a, **_kw):
        await asyncio.sleep(0)
        return next(decisions)

    async def body():
        with patch(
            "app.agents.refinement_loop._generate_critique_decision",
            side_effect=pretend_critique,
        ):
            return await run_phase4_refinement_loop(
                settings=settings,
                corpus=corpus,
                question="topic?",
                max_critique_iterations=3,
                research_runner=tracking_research,
            )

    out = asyncio.run(body())
    assert calls == ["r0", "r1"]
    assert out.terminated_because_satisfied is True
    assert out.final_answer == "DONE-r1"


def test_phase4_exhaustion_sets_flag():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    settings = Settings(gcp_project_id="phase4-exhaust")

    async def noop_research(_s, _c, _prompt: str, slug: str) -> tuple[str, list]:
        await asyncio.sleep(0)
        return slug, []

    async def unsat_always(*_a, **_kw):
        await asyncio.sleep(0)
        return CritiqueDecision(satisfied=False, follow_up_queries=["more"]), "{}"

    async def body():
        with patch(
            "app.agents.refinement_loop._generate_critique_decision",
            side_effect=unsat_always,
        ):
            return await run_phase4_refinement_loop(
                settings=settings,
                corpus=corpus,
                question="q?",
                max_critique_iterations=2,
                research_runner=noop_research,
            )

    out = asyncio.run(body())
    assert out.terminated_because_satisfied is False
    assert len(out.research_history) == 1 + 2
    assert isinstance(out.research_history[0], RefinementResearchRecord)


def test_generate_critique_parses_model_json():
    settings = Settings(gcp_project_id="phase4-json")

    class FakeResp:
        text = (
            '{"satisfied": false, "gaps_identified": ["g"], '
            '"follow_up_queries": ["q"], "reviewer_notes": "n"}'
        )

    generate_fn = AsyncMock(return_value=FakeResp())
    mocked_client = MagicMock()
    mocked_client.aio.models.generate_content = generate_fn

    async def body():
        with patch("google.genai.Client", return_value=mocked_client):
            return await refinement_loop._generate_critique_decision(
                settings,
                original_question="Q?",
                draft_answer="Draft",
                critique_model=None,
                raw_critique_snippet_max=500,
            )

    decision, raw_snip = asyncio.run(body())
    assert decision.satisfied is False
    assert decision.follow_up_queries == ["q"]
    assert generate_fn.await_count == 1
    assert raw_snip


def test_phase4_sync_delegates_to_asyncio_run():
    sentinel = refinement_loop.RefinementLoopResult(
        original_question="q",
        research_history=(),
        critiques=(),
        final_answer="done",
        terminated_because_satisfied=True,
    )
    minimal = {
        "settings": Settings(gcp_project_id="phase4-sync"),
        "corpus": KnowledgeCorpus(FakeEmbeddingBackend(8)),
        "question": "x",
    }

    # ``asyncio.run`` is patched — but Python still validates kwargs before building the coroutine.
    with patch("asyncio.run", return_value=sentinel) as mocked_run:
        out = refinement_loop.run_phase4_refinement_loop_sync(**minimal)

    mocked_run.assert_called_once()
    assert out is sentinel


def test_custom_research_runner_invoked_for_initial_pass():
    recorded: list[str] = []

    async def rr(_s, _c, prompt: str, _slug: str):
        await asyncio.sleep(0)
        recorded.append(prompt.strip())
        return "x", []

    asyncio.run(
        refinement_loop.run_phase4_refinement_loop(
            settings=Settings(gcp_project_id="phase4-inj"),
            corpus=KnowledgeCorpus(FakeEmbeddingBackend(8)),
            question="alpha",
            max_critique_iterations=0,
            research_runner=rr,
        )
    )

    assert recorded == ["alpha"]
