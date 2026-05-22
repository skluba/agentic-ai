"""Phase 2: hybrid private corpus retrieval + Gemini Google Search grounding."""

from __future__ import annotations

from google.adk import Agent

from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool
from app.tools.google_search_tool import make_google_web_search_tool

PHASE_2_SYSTEM_PROMPT = """You are Phase-2 hybrid research answering in **one cohesive reply**.

**Non-negotiable order**
1. **Local-first.** Invoke **`search_private_knowledge`** prior to hosted Google Search\
 whenever it is available.
 Use focused queries; tighten **once** if hits seem thin yet uploads likely hold signal.

**When local grounding returns snippets (`hits` non-empty):**
   - Say **explicitly what local/uploads contain** (faithful summary or tight quotes)\
 before anything else — cite **`chunk_id`** when you quote.
   - **Then extend** via hosted Google Search for gaps: recency, benchmarks, corroboration,\
 global nuance uploads omit.
   - Label extensions visibly (**"From web:"** / **`WEB`**); never silently merge layers.

**When local grounding is empty (`hits` empty) OR the corpus tool is unavailable:**
   - Acknowledge succinctly that **local knowledge lacked relevant snippets**\
 (omit if ingest was empty).
   - Answer **chiefly from hosted Google Search** with **`WEB`** tagging.

Conflicts — uploads anchor internal/product wording. **Web wins** only on clearly external,\
 live-world facts — note **why** briefly when opinions diverge.

Operational guardrails:
- Never hallucinate payloads; summarise only grounded tool outputs.
- If both layers struggle, confess uncertainty plus what verification is missing.
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
        description=(
            "Summarises local-ingest snippets when present; then augments via hosted Gemini search,"
            " else searches the web alone."
        ),
        model=settings.gemini_model,
        instruction=prompt or PHASE_2_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )
