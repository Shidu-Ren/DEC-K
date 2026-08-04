#!/usr/bin/env bash
set -euo pipefail

framework="${1:-core}"
python -m pip install -e .

case "${framework}" in
  core) ;;
  silvr) python -m pip install -r frameworks/silvr/requirements.txt ;;
  m3-agent) (cd frameworks/m3_agent && bash setup.sh) ;;
  all)
    python -m pip install -r frameworks/silvr/requirements.txt
    (cd frameworks/m3_agent && bash setup.sh)
    ;;
  *)
    echo "Usage: bash setup.sh [core|silvr|m3-agent|all]" >&2
    exit 2
    ;;
esac
