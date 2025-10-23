FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY "pyproject.toml" ./

RUN uv pip install -r pyproject.toml --system
COPY "serve_model.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "9696"]
