FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# Install Java and Git LFS
RUN apt-get update && apt-get install -y \
    default-jre wget unzip

# Use buildkit mounts for efficient caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app

RUN wget https://github.com/keeyiahn/egt309-project1/releases/download/datasets/datasets.zip -O datasets.zip && \
    unzip datasets.zip -d data/01_raw && \
    rm datasets.zip

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD [ "uv", "run", "kedro", "run" ]

