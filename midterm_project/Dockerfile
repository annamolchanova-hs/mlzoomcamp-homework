FROM python:3.11-slim

RUN pip install "poetry==1.8.4"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

COPY src/serve.py ./

RUN mkdir artifacts
COPY artifacts/* ./artifacts

EXPOSE 9696

ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]