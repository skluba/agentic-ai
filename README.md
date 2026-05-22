# Agentic AI

Python workspace for evolving a **Gemini 2.0 Flash (Vertex AI)** powered RAG/agent stack with **Google ADK**, **FAISS**, **Langfuse observability**, and **Streamlit** UI patterns. Containers target **GCP project `gd-gcp-gridu-genai`**.

Canonical references:

- [Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A)
- [Langfuse Docker Compose deployment](https://langfuse.com/self-hosting/deployment/docker-compose)
- Langfuse Cursor skill synced from **[github.com/langfuse/skills](https://github.com/langfuse/skills)** into [`.cursor/skills/langfuse/`](./.cursor/skills/langfuse)

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.12 | Runtime + tooling |
| `gcloud auth application-default login` | Authenticate Vertex Gemini calls locally |
| Docker (optional) | UI + infra bundles |
| GitHub Secrets | QA pipeline (`SONAR_TOKEN`) |

### Local bootstrap

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env           # customise secrets / Langfuse URLs
pytest --cov=app --cov-report=term-missing --cov-report=xml
agentic-ai-ui                   # launches Streamlit on :8501
```

### Containers

```bash
docker compose up --build rag-ui        # exposes http://localhost:8501
```

The Compose file mounts **`${HOME}/.config/gcloud` → `/root/.config/gcloud` (read-only)** so **`application_default_credentials.json`** from `gcloud auth application-default login` works inside the container for Vertex Gemini and embeddings (toggle **offline embeddings** in Streamlit around the shared corpus). For the **Gemini Developer API**, set **`GEMINI_API_KEY`** in `.env` (or **`GOOGLE_API_KEY`**) instead; Compose passes **`GEMINI_API_KEY`** through. If mounting ADC is not desired, set **`GOOGLE_APPLICATION_CREDENTIALS`** to a service-account JSON mounted or copied into the image instead.

Self-hosted Langfuse uses the official Compose blueprint (PostgreSQL + Redis + ClickHouse + MinIO)—see **[docs/langfuse-self-hosted.md](docs/langfuse-self-hosted.md)** and the upstream **[Langfuse Compose guide](https://langfuse.com/self-hosting/deployment/docker-compose)**.

## GCP & Langfuse vars

Populate `.env` (values are illustrative):

```
GOOGLE_CLOUD_PROJECT=gd-gcp-gridu-genai
VERTEX_LOCATION=europe-west4
GEMINI_MODEL=gemini-2.0-flash
# Optional Gemini Developer API (AI Studio — used by ADK chat when set)
GEMINI_API_KEY=

LANGFUSE_HOST=http://localhost:3000       # omit to disable instrumentation
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
EMBEDDING_MODEL=text-embedding-004       # Vertex text embeddings
EMBEDDING_DIMENSION=768                  # must match model output dims
```

### Phase 1: Core RAG MVP

End-to-end **Plan → Execute → Synthesize** flow over a **private knowledge base**:

1. **Ingest** — raw text is window-chunked with overlap (`app/knowledge/chunking.py`), embedded (`app/knowledge/embeddings.py`), L2-normalized, and indexed in **FAISS** (`app/knowledge/store.py`).
2. **Execute** — the ADK agent `core_rag` (`app/agents/core_rag.py`) must call **`search_private_knowledge`** (`app/tools/document_search_tool.py`) before answering.
3. **Synthesize** — the model summarizes grounded snippets into a concise reply.

Try it in Streamlit (**Phase 1 RAG** tab): ingest sample notes in the **shared corpus** strip, then switch to the Phase 1 tab — use **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) for the AI Studio Gemini path or Vertex ADC **`gcloud auth application-default login`** for enterprise routing. Programmatic helper: `run_core_rag_turn_sync` / `run_core_rag_turn` in `app/agents/session_runner.py`.

### Phase 2: External knowledge (Hybrid)

**Corpus-first:** optional uploads stay the default authority; **hosted Google Search** is for gaps, corroboration, or explicitly live/global questions (`app/agents/external_knowledge.py`).

1. **Planner-aware execution** — calls **`search_private_knowledge`** before web when chunks exist unless the prompt is inherently web-only or the corpus is empty.
2. **Synthesis** — **`PRIVATE_KB`** vs **`WEB`** tags; internal conflicts favour **`PRIVATE_KB`** unless the question is time-sensitive worldly fact.
3. **Streamlit — Phase 2 tab** — shares the ingest strip with Phase 1 (`uploads` optional; grounding still available).

Use **`run_phase2_external_turn_sync`** / **`run_phase2_external_turn`** from **`app/agents/session_runner.py`**.

---

## QA + SonarCloud

The [`QA` workflow](./.github/workflows/qa.yml) runs **formatting (Ruff)**, **lint (Ruff)**, **pytest with coverage**, and **SonarCloud** using `sonar-project.properties`.

Setup checklist:

1. Import the repo (or recreate a project manually) inside [SonarCloud](https://sonarcloud.io) under organisation **`skluba`** using key **`skluba_agentic-ai`** (matching `sonar-project.properties`).
2. Generate a **`SONAR_TOKEN`** with analyse permissions and save it under **Repo → Settings → Secrets → Actions**.
3. Re-run workflow after the secret is saved.

Optional: SonarLint / IDE connected mode uses [`.sonarlint/connectedMode.json`](./.sonarlint/connectedMode.json).

---

## Repo map

```
app/config.py               # pydantic-settings for Vertex + Langfuse + embeddings
app/knowledge/              # Phase 1 chunk → embed → FAISS corpus + search_chunks
app/agents/core_rag.py      # Phase 1 Plan/Execute/Synthesize + document tool
app/agents/external_knowledge.py # Phase 2 hybrid planner + corpus + Google Search grounding
app/agents/session_runner.py
app/tools/document_search_tool.py
app/tools/google_search_tool.py
app/rag/faiss_store.py      # deterministic in-memory retrieval slice (lab demo)
app/rag/lab_demo.py         # hierarchical @observe (chain + retriever spans)
app/observability/          # Langfuse helpers + flush for scripts / Streamlit
streamlit_app.py            # Loads .env first; Smoke + Phase 1 RAG tabs
.cursor/skills/langfuse/    # upstream Langfuse agent skill (+ references/)
infra / docs forthcoming    # richer ADK graphs + A2A wiring live here next
```

## License

Apache-2.0 (mirror external dependencies licensing as needed).
