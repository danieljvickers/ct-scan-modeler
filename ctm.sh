#!/usr/bin/env bash
# entry point: ./ctm.sh <run|build|test|model> [options], see ./ctm.sh --help

CTM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$CTM_ROOT/build"
. "$CTM_ROOT/toolchain/bootstrap/python.sh"

PYTHONUNBUFFERED=1 PYTHONPATH="$CTM_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$CTM_ROOT/build/venv/bin/python" -m toolchain.main "$@"
