# Manual testing (Streamlit RAG lab)

Use this playbook to exercise **Phase 1–5** tabs in [`streamlit_app.py`](../streamlit_app.py) end-to-end. Example paste corpora live in [`examples/`](examples/).

## Prerequisites

| Need | Purpose |
|------|---------|
| Python env from repo root | `pip install -e ".[dev,phase3-fetch,news-agent]"` ([README](../README.md)) |
| Gemini / Vertex | Phases **1**, **2**, **4**, **5**: Application Default Credentials and `GOOGLE_CLOUD_PROJECT` + region. Optional `GEMINI_API_KEY` fallback. |
| Langfuse keys (optional) | Tracing on **Instrumentation**. For **`rag-ui` in Docker**, **`LANGFUSE_HOST=http://localhost:3000`** hits the container — use **`http://host.docker.internal:3000`** if Langfuse runs on the host (Docker Desktop), or omit keys. |
| **Phase 3** | Yahoo MCP **`fetch`** (see [README](../README.md)). **`rag-ui`** Compose pins **`python -m`**; host `.env` **`docker`** is ignored there. |
| **Phase 5** | Standalone **News Agent** on **:8090**; **`docker compose`** collaboration profile pins internal URLs (see [`.env.example`](../.env.example)). |

## Two-process caveat (Phase 5 corpus)

| Where you ingest | What it affects |
|------------------|-----------------|
| **Streamlit · shared corpus strip** | FAISS for **`rag-ui`** only — Phase **1**, **2**, **3**, **4**, **5** orchestrator. |
| **`news-agent` container / uvicorn process** | Starts with an **empty** [`KnowledgeCorpus`](../app/knowledge/store.py) unless you extend deployment to load text. |

So **`delegate_to_news_kb_specialist_via_a2a`** often returns briefing sections labeled **`NEWS_AGENT · WEB`** (web-first) while the orchestrator still uses your Streamlit ingest for **`search_private_knowledge`**. That split is expected in the default Compose stack.

## Shared corpus strip (first step for Phases 1–2, 4–5 orchestrator)

1. Enter a stable **human-readable corpus label**, e.g. `manual-demo-acme-gridu`.
2. Open and copy **[`examples/corpus_acme_internal_memo.txt`](examples/corpus_acme_internal_memo.txt)** into **Paste knowledge-base text**, then click **Ingest corpus**.
3. Optionally append **[`examples/corpus_gridu_research_lab.txt`](examples/corpus_gridu_research_lab.txt)** and/or **[`examples/corpus_newsroom_wire.txt`](examples/corpus_newsroom_wire.txt)** (Phase **5** policy-style stub) and ingest again.
4. Confirm success toasts (“Ingested … chunks”; Session ID shown).

Suggested **doc identifiers** in examples: `MEMO-2026-ORG-0421`, `NEWSROOM-WIRE-2026-01`.

**Copy/paste Phase 5 orchestrator prompts** (optional): [`examples/phase5_copy_paste_prompts.txt`](examples/phase5_copy_paste_prompts.txt).

## Quick start: Phase 5 with Docker

```bash
docker compose --profile collaboration up --build rag-ui news-agent
```

- UI: `http://localhost:8501` — open **Phase 5 · A2A collaborative news**.
- News Agent health: `GET http://localhost:8090/.well-known/agent-card.json` with header `A2A-Version: 1.0` (expect JSON agent card).
- **Orchestrator + agent card base:** `docker-compose.yml` pins **`NEWS_AGENT_A2A_BASE_URL`** and **`NEWS_AGENT_PUBLIC_BASE_URL`** to **`http://news-agent:8090`** on both services (so A2A follow-up calls from **`rag-ui`** are not sent to **`localhost`**). You can still probe health from the host at **`http://localhost:8090`** via port mapping.

### Troubleshooting Phase 5

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| `Network communication error ... news-agent ... [Errno -2] Name or service not known` | **A)** Streamlit on the **host** with **`NEWS_AGENT_A2A_BASE_URL=http://news-agent:8090`** — that hostname exists only on Compose DNS. **B)** **`news-agent` container is not Up** (often **Exited (2)**): the image used **`ENTRYPOINT streamlit`** while Compose passed **`uvicorn`**, so Uvicorn never bound to 8090 (fixed in current **`Dockerfile`**: empty `ENTRYPOINT`, Streamlit as **`CMD`** only). | **A)** **`NEWS_AGENT_A2A_BASE_URL=http://localhost:8090`** in `.env`; restart host Streamlit. **B)** Run **`docker compose --profile collaboration ps`** — **`news-agent`** must be **Up**. Rebuild: **`docker compose --profile collaboration build --no-cache news-agent`** then **`up -d`**; check logs: **`docker compose logs news-agent`** (expect Uvicorn listening, not Streamlit). |
| `Network communication error ... All connection attempts failed` (after **`GET /.well-known/agent-card.json` succeeds from `rag-ui`) | Host `.env` set **`NEWS_AGENT_PUBLIC_BASE_URL=http://localhost:8090`** so **`AgentCard.supported_interfaces[].url`** pointed at **`localhost`**; from **`rag-ui`**, **localhost** is the Streamlit container, so follow-up RPC never reaches **`news-agent`**. Compose now pins **`NEWS_AGENT_PUBLIC_BASE_URL=http://news-agent:8090`** on the **`news-agent`** service (`docker-compose.yml`). | **`docker compose up -d news-agent`** (or **`--force-recreate`**) after pulling compose changes; **`docker compose logs`** should show follow-up **`POST`** traffic from **`172.*`** peers. |

## Tab-by-tab checklist

### Instrumentation (smoke only)

**Goal**: Langfuse client initializes without raising.

**Expected**: Green success path if keys set; graceful message if tracing disabled/misconfigured — no uncaught traceback.

---

### Phase 1 — Core RAG

**Depends on**: Ingest completed for your corpus label.

| Example question | What good looks like |
|------------------|---------------------|
| “What retention period applies to support transcripts in the memo?” | Answer cites ~18 months; grounded in corpus, not hallucinated vendors. |
| “What mitigation fixed duplicate chunks for ticket NW-8891?” | Mentions v2.3.1 / release note in memo. |
| “Who is the pipeline owner and where do they hang out?” | Platform team; #platform-ingest (if present in pasted text). |

---

### Phase 2 — Hybrid

**Depends on**: Same ingest; model + search available.

| Example question | What good looks like |
|------------------|---------------------|
| “Summarize the ACME retention section and add one external best practice for log retention.” | Mix of corpus + clearly separated web supplement. |
| “What is Langfuse used for in this lab’s notes?” | Pulls from second corpus file if ingested. |

---

### Phase 3 — MCP Yahoo (demo)

**Depends on**: Yahoo MCP running; tool enabled in UI.

| Example question | What good looks like |
|------------------|---------------------|
| “What was the most recent closing price you can retrieve for AAPL?” | Tool-backed numbers; may vary by market day. |
| “Compare AAPL and MSFT year-to-date return if the tool exposes it.” | Uses finance tool; explicit if data missing. |

If MCP is down: expect **error path** with a clear message — log as defect only if the app crashes.

---

### Phase 4 — Refinement loop

**Depends on**: Ingest + hybrid path working.

| Example question | What good looks like |
|------------------|---------------------|
| “Using only our internal memo, what customer-facing promise must we avoid?” | Iteration improves citation / avoids “instant” wording from memo. |
| “List open questions from the lab notes and whether the memo answers them.” | Grounded comparison across ingested snippets. |

---

### Phase 5 — A2A collaborative news

**Depends on**: Same Streamlit ingest as above (orchestrator); **`NEWS_AGENT_A2A_BASE_URL`** set; **News Agent** process reachable.

| Scenario | Example question | What good looks like |
|----------|------------------|---------------------|
| News delegation | “Give a concise briefing of major EU tech-policy headlines from the last week with dates where possible.” | Model uses **`REMOTE_NEWS_AGENT`** (or equivalent) for the delegated digest; no hard crash if News Agent is slow (watch timeout). |
| Corpus + remote | (Ingest **newsroom wire** + memo) “Summarize recent AI regulatory headlines and apply our **[ACME-CHECK]** / wire rules.” | Orchestrator cites internal chunks; delegated block labeled as remote specialist output. |
| MCP + News | “Show Yahoo most-actives via the finance tool, then one paragraph of sector news context.” | MCP table plus optional A2A news paragraph. |
| Config off | Clear **`NEWS_AGENT_A2A_BASE_URL`** → reload app | Warning in tab; Phase **5** run still engages Phase **3**-class tools **without** the delegation tool. |

## Files in `examples/`

| File | Role |
|------|------|
| [`corpus_acme_internal_memo.txt`](examples/corpus_acme_internal_memo.txt) | Dense factual memo (retention, release, contacts). |
| [`corpus_gridu_research_lab.txt`](examples/corpus_gridu_research_lab.txt) | Short second doc for hybrid / multi-topic behavior. |
| [`corpus_newsroom_wire.txt`](examples/corpus_newsroom_wire.txt) | Editorial-style stub for Phase **5** orchestrator-side policy tests. |
| [`phase5_copy_paste_prompts.txt`](examples/phase5_copy_paste_prompts.txt) | Ready-made Phase **5** questions (duplicate of suggestions above). |

## Reporting issues

Note: corpus label used, Gemini model ID, MCP up/down, **`NEWS_AGENT_*` values**, whether **News Agent** responded at `8090`, and **short repro question**. Attach Langfuse trace ID if instrumentation was on.
