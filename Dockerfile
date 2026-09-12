FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY *.py .
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ tests/

CMD ["python", "train.py", "--timesteps", "200000"]
