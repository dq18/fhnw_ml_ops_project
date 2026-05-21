FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV APP_DIR="/app"
WORKDIR $APP_DIR
ENV VIRTUAL_ENV="$APP_DIR/.venv"
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

ARG UV_VERSION=0.9.26
ENV UV_NO_CACHE=1
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV
RUN curl -Lsf --proto '=https' https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# Install dependencies (cached layer)
COPY pyproject.toml ./
RUN uv venv $VIRTUAL_ENV && uv pip install -r pyproject.toml

# Copy source code and data
COPY src/ src/
COPY data/ data/
COPY config.json .
COPY model_config.json .
# NOTE: do NOT bake .env into the image.
# Pass secrets at runtime: docker run --env-file .env ...

# Expose Streamlit port (only used when running the app container)
EXPOSE 8501

# Default: run the feature pipeline
# Override with:  docker run ... python -m src.pipelines.training_pipeline
#                 docker run ... streamlit run src/app/streamlit_app.py --server.port 8501
CMD ["python", "-m", "src.pipelines.feature_pipeline"]
