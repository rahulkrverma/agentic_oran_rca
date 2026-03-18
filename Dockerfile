FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agentic_oran_rca /app/agentic_oran_rca

EXPOSE 8000

CMD ["python", "-m", "agentic_oran_rca.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
