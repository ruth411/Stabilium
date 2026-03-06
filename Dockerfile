FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi>=0.110 "uvicorn[standard]>=0.29" pydantic>=2.7

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
