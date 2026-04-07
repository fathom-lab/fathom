FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx pydantic
COPY fathom_api_proxy.py .
CMD ["uvicorn", "fathom_api_proxy:app", "--host", "0.0.0.0", "--port", "8080"]
