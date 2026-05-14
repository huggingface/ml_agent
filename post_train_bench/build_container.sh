#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash post_train_bench/build_container.sh [--image IMAGE] [--sqsh-output PATH]

Build the PostTrainBench solve/judge Docker image. When --sqsh-output is set,
also import the local Docker image into an Enroot squashfs file for Pyxis.
EOF
}

IMAGE="${POST_TRAIN_BENCH_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest}"
SQSH_OUTPUT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift
            ;;
        --sqsh-output)
            SQSH_OUTPUT="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

docker build -t "$IMAGE" -f post_train_bench/Dockerfile .

if [ -n "$SQSH_OUTPUT" ]; then
    if [ -e "$SQSH_OUTPUT" ]; then
        echo "Refusing to overwrite existing squashfs: $SQSH_OUTPUT" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$SQSH_OUTPUT")"
    ENROOT_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/enroot-${USER:-user}"
    export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${ENROOT_BASE}/cache}"
    export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${ENROOT_BASE}/data}"
    export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${ENROOT_BASE}/runtime}"
    mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
    enroot import --output "$SQSH_OUTPUT" "dockerd://${IMAGE}"
fi
