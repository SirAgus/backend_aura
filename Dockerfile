
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

WORKDIR /app

# Copy configuration files
COPY pyproject.toml pdm.lock README.md ./

# Install dependencies
RUN pdm install --prod --no-lock --no-editable

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["pdm", "run", "start"]
