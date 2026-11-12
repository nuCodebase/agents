FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY docker/fastapi-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_api.py agents_strands.py agent_logging.py plot_agent_timeline.py ./
COPY tools ./tools
COPY mcp_servers ./mcp_servers

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
