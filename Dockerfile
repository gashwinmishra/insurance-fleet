FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Create non-root user for container security
RUN groupadd --system appgroup && useradd --system --gid appgroup --create-home appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/temp_uploads && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/_stcore/health')" || exit 1

CMD ["streamlit", "run", "ui/dashboard.py", "--server.port=8080", "--server.enableCORS=true", "--server.enableXsrfProtection=true"]

