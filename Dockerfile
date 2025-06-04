FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app

RUN apt-get update && apt-get install -y default-jre
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD [ "uv", "run", "kedro", "run" ]
