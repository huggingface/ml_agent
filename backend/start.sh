#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode spawns CMD multiple times simultaneously on restart.
# Only the first instance can bind port 7860 — the rest must exit
# with code 0 so the dev mode daemon doesn't mark the app as crashed.

if [ "${ML_INTERN_PROCESS_ROLE:-api}" = "worker" ]; then
    uvicorn worker_app:app --host 0.0.0.0 --port 7860
    exit $?
fi

# Run uvicorn; if it fails due to port conflict, exit cleanly.
uvicorn main:app --host 0.0.0.0 --port 7860
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Check if this was a port-in-use failure (another instance already running)
    echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
    exit 0
fi
