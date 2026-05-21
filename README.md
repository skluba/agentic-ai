# Agentic AI

Python workspace for evolving a **Gemini 2.0 Flash (Vertex AI)** powered RAG/agent stack with **Google ADK**, **FAISS**, **Langfuse observability**, and **Streamlit** UI patterns. Containers target **GCP project `gd-gcp-gridu-genai`**.

Canonical references:

- [Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A)
- [Langfuse Docker Compose deployment](https://langfuse.com/self-hosting/deployment/docker-compose)

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

Self-hosted Langfuse uses the official Compose blueprint (PostgreSQL + Redis + ClickHouse + MinIO)—see **[docs/langfuse-self-hosted.md](docs/langfuse-self-hosted.md)** and the upstream **[Langfuse Compose guide](https://langfuse.com/self-hosting/deployment/docker-compose)**.

## GCP & Langfuse vars

Populate `.env` (values are illustrative):

```
GOOGLE_CLOUD_PROJECT=gd-gcp-gridu-genai
VERTEX_LOCATION=europe-west4
GEMINI_MODEL=gemini-2.0-flash
LANGFUSE_HOST=http://localhost:3000       # omit to disable instrumentation
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

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
app/config.py               # pydantic-settings for Vertex + Langfuse
app/rag/faiss_store.py      # deterministic in-memory retrieval slice
app/observability/          # Langfuse client helpers
streamlit_app.py            # Thin UI façade
infra / docs forthcoming    # richer ADK graphs + A2A wiring live here next
```

## License

Apache-2.0 (mirror external dependencies licensing as needed).
