FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDEX_FILE=/data/index.json \
    TICKET_DB_FILE=/data/tickets.sqlite3

WORKDIR /app

# The dev requirements include pytest, so the tests can run inside the image.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY main.py prompts.py test_main.py ./

RUN useradd --create-home app && mkdir /data && chown app:app /data
USER app
VOLUME /data
EXPOSE 8000

# The first start crawls the site before serving, so allow a long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
