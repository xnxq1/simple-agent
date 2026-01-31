FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-install-project

# Copy source code
COPY . .

# Sync project dependencies
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "manage.py", "start-api"]