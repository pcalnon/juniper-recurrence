# syntax=docker/dockerfile:1
# =============================================================================
# juniper-recurrence — Δt-native LMU recurrence FastAPI + CLI service
# -----------------------------------------------------------------------------
# Multi-stage image (roadmap OUT-4 / WS-7). The LMU memory is a fixed linear
# recurrence (numpy-only — torch is a deferred increment), so unlike the cascor
# worker this image needs no torch / CUDA dance and stays slim by nature. All
# upstream juniper packages are published to PyPI, so the build installs from
# PyPI with no editable-sibling / cross-repo-clone step.
#
# Build context is the app subdirectory (this repo is a monorepo: the app lives
# in juniper-recurrence/, the model in juniper-recurrence-model/).
#
#   Build: docker build -t juniper-recurrence:latest juniper-recurrence/
#   Run:   docker run -p 8211:8210 juniper-recurrence:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder — install the app + deps into a clean site-packages
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Dependencies come from requirements.lock, NOT from a bare `pip install ".[extra]"`.
# Unpinned, the image's contents were whatever PyPI served at build time: two builds of
# the same commit could ship different versions, and a bad upstream release reached the
# image with nothing to stop it. The lock's header carries the regeneration recipe and
# the arm64 wheel-availability pre-flight this image depends on (no gcc, no Rust here, so
# a pin with no aarch64 wheel is a hard build failure on that arch).
#
# Copied and installed BEFORE the app source so a source-only change reuses this layer.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy only what the build needs (pyproject reads README.md; version is a static
# attr in juniper_recurrence/_version.py, so no git / setuptools-scm required).
COPY pyproject.toml README.md ./
COPY juniper_recurrence/ ./juniper_recurrence/

# The app itself, with deps already satisfied above. --no-deps keeps pip from
# re-resolving anything the lock pinned; `pip check` then proves the installed set is
# mutually consistent, so a lock that has drifted from pyproject.toml (a dependency
# added but not re-locked) fails HERE, at build time, rather than at import on a Pi.
# The [observability] extra's contents are in the lock -- it is compiled with
# `--extra observability` -- so installing without the extra marker loses nothing.
RUN pip install --no-cache-dir --no-deps . && pip check

# -----------------------------------------------------------------------------
# Stage 2: Runtime — minimal production image
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
# stamped as OCI labels so `docker inspect` / the deploy Makefile can detect
# stale-image drift. Default empty when built bare.
ARG GIT_SHA=""
ARG BUILD_DATE=""
ARG APP_VERSION=""

LABEL org.opencontainers.image.title="juniper-recurrence"
LABEL org.opencontainers.image.description="Δt-native LMU recurrence FastAPI + CLI service"
LABEL org.opencontainers.image.authors="Paul Calnon"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/pcalnon/juniper-recurrence"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.version="${APP_VERSION}"

# Also expose provenance to the running process (OBS-03): juniper_recurrence.provenance
# reads these back so set_build_info() can populate the juniper_recurrence_build_info
# Info metric. Empty outside a provenance-stamped build (provenance.* then return None).
ENV JUNIPER_RECURRENCE_GIT_SHA=${GIT_SHA}
ENV JUNIPER_RECURRENCE_BUILD_DATE=${BUILD_DATE}

# Non-root runtime user.
RUN groupadd --gid 1000 juniper && \
    useradd --uid 1000 --gid juniper --shell /bin/bash --create-home juniper

WORKDIR /app

# Copy the installed packages + console scripts from the builder.
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p logs && chown -R juniper:juniper /app

USER juniper

# Service configuration (overridden by Docker Compose). Container port 8210;
# the deploy compose maps host 8211 -> container 8210 (design §6.8 / OQ-15).
ENV JUNIPER_RECURRENCE_HOST=0.0.0.0 \
    JUNIPER_RECURRENCE_PORT=8210

EXPOSE 8210

# HTTP health check (the app is an HTTP server, unlike the worker's WS client).
# /v1/health is provided by juniper-service-core and exempt from the API-key
# auth middleware, so it answers 200 even on a secured image.
# start-period=40s: the app imports a heavy pure-Python stack (service-core,
# model-core, recurrence-model, data-client, numpy, fastapi) — ~10-15s — before
# uvicorn binds, so early failing probes must not mark the container unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8210/v1/health', timeout=5).status == 200 else 1)"

ENTRYPOINT ["juniper-recurrence"]
CMD ["serve"]
