#!/usr/bin/env bash
# benchmark-opt/build-sandbox-image.sh
#
# Builds the Docker sandbox image used by run-agent-bench-docker.sh for codex runs.
# The image extends the base Neuron PyTorch image with Node.js, npm, and codex.
#
# Usage:
#   ./benchmark-opt/build-sandbox-image.sh [--tag TAG]
#
# Options:
#   --tag TAG   Image tag to build (default: nki-bench-sandbox:latest)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="nki-bench-sandbox:latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building sandbox image: $IMAGE_TAG"
echo "Dockerfile: $SCRIPT_DIR/Dockerfile.sandbox"
echo ""

docker build \
    -t "$IMAGE_TAG" \
    -f "$SCRIPT_DIR/Dockerfile.sandbox" \
    "$SCRIPT_DIR"

echo ""
echo "Done. Image built: $IMAGE_TAG"
