# ==========================================
# Global ARGs
# ==========================================
ARG DOCKER_REGISTRY=
ARG GHCR_REGISTRY=ghcr.io/

# ==========================================
# Stage 1: Build Frontend
# ==========================================
FROM ${DOCKER_REGISTRY:-}node:18-alpine AS frontend-builder

ARG NPM_REGISTRY=

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN if [ -n "$NPM_REGISTRY" ]; then \
        npm config set registry "$NPM_REGISTRY"; \
    fi && \
    (npm install --frozen-lockfile || npm install)

# Copy source code
COPY frontend/ ./

# Build frontend
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ==========================================
# Stage 2: Build Backend Environment (uv)
# ==========================================
FROM ${GHCR_REGISTRY}astral-sh/uv:latest AS uv

# ==========================================
# Stage 3: Final Image
# ==========================================
FROM ${DOCKER_REGISTRY:-}python:3.10-slim

ARG APT_MIRROR=
ARG PYPI_INDEX_URL=

WORKDIR /app

# Install system dependencies
RUN if [ -n "$APT_MIRROR" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i "s@deb.debian.org@$APT_MIRROR@g" /etc/apt/sources.list.d/debian.sources; \
        else \
            echo "Warning: /etc/apt/sources.list.d/debian.sources not found, skipping mirror setup." >&2; \
        fi; \
    fi && \
    apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=uv /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Copy backend requirements
COPY pyproject.toml uv.lock* ./

# Configure PyPI mirror
ENV UV_INDEX_URL=${PYPI_INDEX_URL}
ENV UV_HTTP_TIMEOUT=300

# Install Python dependencies
RUN if [ -f uv.lock ]; then \
        uv sync --frozen; \
    else \
        uv sync; \
    fi

# Copy backend code
COPY backend/ ./backend/

# Copy frontend build artifacts to backend static folder
# backend/app.py expects them at backend/static/dist
COPY --from=frontend-builder /app/frontend/dist ./backend/static/dist

# Create necessary directories
RUN mkdir -p /app/backend/instance /app/uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=backend/app.py
# PORT defaults to 5000, can be overridden
ENV PORT=5000 
# Tell backend it's running in Docker
ENV IN_DOCKER=1

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["sh", "-c", "curl -f http://localhost:${PORT:-5000}/health || exit 1"]

# Start command
# 1. Run database migrations
# 2. Start Flask app
CMD ["sh", "-c", "uv run --directory backend alembic upgrade head && uv run --directory backend python app.py"]
