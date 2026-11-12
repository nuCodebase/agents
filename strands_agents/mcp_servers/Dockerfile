FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY mcp_servers/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_servers ./mcp_servers

EXPOSE 8000

RUN useradd --create-home --uid 10001 mcpuser
USER mcpuser

CMD ["python", "-m", "mcp_servers.math_server"]
