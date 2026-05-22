FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

COPY pyproject.toml README.md LICENSE streamlit_app.py ./
COPY app ./app

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[phase3-fetch,news-agent]"

EXPOSE 8501 8090

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
