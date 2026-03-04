# -------------------------------------------------------------------
# STAGE 1: BUILDER
# -------------------------------------------------------------------
FROM python:3.11-slim as builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements
COPY requirements/frontend.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r frontend.txt

# -------------------------------------------------------------------
# STAGE 2: RUNTIME
# -------------------------------------------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

RUN mkdir -p /home/appuser && chown -R appuser:appgroup /home/appuser

# Copy virtual env
COPY --from=builder /app/.venv /app/.venv

# Update PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV HOME=/home/appuser

# Copy source code
COPY --chown=appuser:appgroup src/frontend/ ./src/frontend/

# Ensure /app ownership is correct
RUN chown -R appuser:appgroup /app

# Switch user
USER appuser

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "src/frontend/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]