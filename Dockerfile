# Multi-stage build for DataSentience backend
# Stage 1: Builder - Install dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and README.md for dependency installation
COPY pyproject.toml README.md ./

# Install Python dependencies using pip with pyproject.toml
RUN pip install --no-cache-dir --user -e .

# Stage 2: Runtime - Minimal production image
FROM python:3.13-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Copy environment configuration
COPY .env.example .env

# Copy SageMaker entrypoint script
COPY entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create data output directories
RUN mkdir -p data/logs data/manuals

# Expose port (8080 required for SageMaker)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# SageMaker entrypoint
ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]

# Default command for SageMaker
CMD ["serve"]