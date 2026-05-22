# Manual testing (Streamlit RAG lab)

Use this playbook to exercise **Phase 1–4** tabs in [`streamlit_app.py`](../streamlit_app.py) end-to-end. Example paste corpora live in [`examples/`](examples/).

## Prerequisites

| Need | Purpose |
|------|---------|
| Python env from repo root | `uv sync` |
| Gemini / Vertex | Phases **1**, **2**, **4**: Application Default Credentials and `GOOGLE_CLOUD_PROJECT` + `REGION` ([README](../README.md)). Optional `GEMINI_API_KEY` fallback. |
| Langfuse keys (optional) | Tracing smoke test on **Instrumentation**. |
| **Phase 3** | **Finance MCP demo** expects the Yahoo MCP server per [README bootstrap](../README.md) (`yahoo-finance`). Without it, expect Phase 3 fetch failures. |

## Shared corpus strip (first step for Phases 1–2 and 4)

1. Enter a stable **human-readable corpus label**, e.g. `manual-demo-acme-gridu`.
2. Open and copy the contents of **[`examples/corpus_acme_internal_memo.txt`](examples/corpus_acme_internal_memo.txt)** into **Paste knowledge-base text**, then click **Ingest corpus**.
3. Optionally append **[`examples/corpus_gridu_research_lab.txt`](examples/corpus_gridu_research_lab.txt)** into the same text area and ingest again **or** use a second ingest after editing the paste buffer (depending on whether you want one combined blob or sequential chunks — both are fine for QA).
4. Confirm success toasts (“Ingested … chunks”; Session ID shown).

Suggested **doc identifiers** referenced in examples below: `MEMO-2026-ORG-0421` (memo), `GEMINI` / `Langfuse` (lab notes).

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

## Files in `examples/`

| File | Role |
|------|------|
| [`corpus_acme_internal_memo.txt`](examples/corpus_acme_internal_memo.txt) | Dense factual memo (retention, release, contacts). |
| [`corpus_gridu_research_lab.txt`](examples/corpus_gridu_research_lab.txt) | Short second doc for hybrid / multi-topic behavior. |

## Reporting issues

Note: corpus label used, Gemini model ID, whether Phase 3 MCP was up, and **short repro question**. Attach Langfuse trace ID if instrumentation was on.
