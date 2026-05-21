# Langfuse (self-hosted) companion notes

Production-grade Langfuse requires several backing services (PostgreSQL, Redis, ClickHouse, MinIO, workers). Follow Langfuse's maintained Compose blueprint:

```
https://langfuse.com/self-hosting/deployment/docker-compose
```

Suggested workflow:

1. Clone upstream `langfuse/langfuse`.
2. Copy `docker-compose.yml` + `.env.example` from that repo and regenerate every secret marked **`CHANGEME`** (`openssl rand -hex 32`).
3. Expose `:3000` only on trusted interfaces; tighten firewall rules (`langfuse-worker`, MinIO consoles, ClickHouse endpoints).
4. Create API ingestion keys (`pk-lf-…`, `sk-lf-…`) inside the Langfuse UI and mirror them inside this app's `.env` (`LANGFUSE_*` vars).

Monitoring path for ADK/Python:

- Import `observe` decorators from Langfuse SDK (preferred) wherever graph nodes emit traces.
- For rapid Streamlit prototyping, call `start_optional_span` from `app.observability`.
