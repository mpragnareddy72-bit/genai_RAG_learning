FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv

COPY backend ./backend
COPY src ./src

RUN uv pip install --system .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t rag-backend -f backend/Dockerfile .