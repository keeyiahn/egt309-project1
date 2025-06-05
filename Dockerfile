FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# Install Java and Git LFS
RUN apt-get update && apt-get install -y \
    default-jre \
    git \
    curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install

# Use buildkit mounts for efficient caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app

# Pull Git LFS files
RUN git lfs pull

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD [ "uv", "run", "kedro", "run" ]

