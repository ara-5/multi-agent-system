FROM python:3.11-slim

# uv gives a reproducible install from the committed lockfile (uv.lock) --
# everyone building this image gets identical dependency versions, not just
# versions satisfying pyproject.toml's ranges -- and installs markedly faster
# than pip, which matters for CI image-build time.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .
RUN uv sync --extra dev --frozen

CMD ["uv", "run", "spread/train.py", "--timesteps", "200000"]
