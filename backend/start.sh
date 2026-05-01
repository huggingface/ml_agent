#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode spawns CMD multiple times simultaneously on restart.
# Only the first instance can bind port 7860 — the rest must exit
# with code 0 so the dev mode daemon doesn't mark the app as crashed.

# Worker mode: no HTTP listener, no port binding — just run the
# session-claim loop forever. Invoked from WORKDIR=/app/backend so the
# module path matches the existing `uvicorn main:app` style below.
if [ "$MODE" = "worker" ]; then
    exec python -m worker
fi

# Main mode (default): existing port-conflict graceful behavior.
uvicorn main:app --host 0.0.0.0 --port 7860
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Check if this was a port-in-use failure (another instance already running)
    echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
    exit 0
fi
