FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_RETRIES=10
ENV PIP_DEFAULT_TIMEOUT=300

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir --retries 10 --timeout 300 -r requirements.txt

COPY backend ./backend
COPY ml ./ml
COPY artifacts ./artifacts
COPY feature_store ./feature_store

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]