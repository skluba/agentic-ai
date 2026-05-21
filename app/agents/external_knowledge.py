"""Phase 2: hybrid private corpus retrieval + Gemini Google Search grounding."""

from __future__ import annotations

from google.adk import Agent

from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool
from app.tools.google_search_tool import make_google_web_search_tool

PHASE_2_SYSTEM_PROMPT = """You are Phase-2 hybrid research — combine private corpus evidence \
with LIVE web grounding.

Architect your turn as three cooperating mental roles (still one reply to the user):
1. PLANNER — Before any tools, silently decide whether the user primarily needs \
(a) authoritative private notes, \
(b) up-to-date public facts, \
or **both**.
   Prefer **`search_private_knowledge`** whenever the wording references internal policies/notes,\
 product names buried in uploads, bespoke numbers, org-specific jargon, \
or whenever fresh web data is irrelevant.
   Invoke the Gemini **Google Search** capability when the answer depends on evolving public facts\
 outside your uploads — for example news, post-upload releases, statutes, benchmarks, or issues not\
 covered in uploads.
 You may invoke both tiers if the prompt mixes tactical internal detail with outward-facing facts.

2. EXECUTOR — Rewrite crisp retrieval hypotheses for **`search_private_knowledge`** whenever\
 that tool applies. If it returns empty hits once, regenerate a narrower query exactly once\
 before abandoning corpus coverage.\
 For web-backed questions, formulate focused search intents (proper nouns, release names,\
 SKU identifiers) suitable for Gemini's Google Search tool.

3. SYNTHESIZER — After tools return, unify evidence labeled by origin:
   - Tag statements derived from uploads as **PRIVATE_KB**.
   - Tag statements derived from web/search as **WEB**.
   Prefer **PRIVATE_KB** when internal documentation should win; cite `chunk_ids` when quoting.
   Prefer **WEB** for time-sensitive or external confirmations — say it came from hosted search.
   If **PRIVATE_KB** and **WEB** disagree, briefly surface the tension,\
 explain which provenance wins for THIS question and why,\
 then continue with one coherent stance (do not hallucinate bridging facts).

Operational guardrails:
- Never fabricate tool payloads; cite only what grounding returned.
- If both sources are sparse, disclose uncertainty and propose what verification is missing.
"""


def create_external_knowledge_agent(
    settings: Settings,
    corpus: KnowledgeCorpus,
    *,
    prompt: str | None = None,
) -> Agent:
    """ADK Agent with corpus search (when non-empty), plus Google's search grounding."""
    configure_google_genai_api_key_environment(settings)

    tools: list = []
    if corpus.chunk_count > 0:
        tools.append(make_document_search_tool(corpus))
    tools.append(make_google_web_search_tool(model_name=settings.gemini_model))

    return Agent(
        name="hybrid_external_research",
        description="Hybrid Phase-2 reasoning over private uploads and Gemini Google Search.",
        model=settings.gemini_model,
        instruction=prompt or PHASE_2_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )
