FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi>=0.110 "uvicorn[standard]>=0.29" pydantic>=2.7

EXPOSE 8000

CMD ["sh", "./start.sh"]
