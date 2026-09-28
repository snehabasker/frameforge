#!/usr/bin/env bash
# Generate a CPU flamegraph for the FrameForge engine (CLAUDE.md 7.5).
#
# Requires Linux `perf` and Brendan Gregg's FlameGraph scripts
# (https://github.com/brendangregg/FlameGraph, CDDL-1.0, see THIRD_PARTY.md
# -- cloned by this script into a local ignored dir, not vendored, since
# the scripts themselves are never linked into any shipped binary).
#
# NOT runnable on this project's Windows dev machine (no perf) -- authored
# and documented for the Linux CI/dev environment where it's meant to run.
# See docs/DECISIONS.md and docs/ARCHITECTURE.md's "What's not built yet".
#
# Usage:
#   ./scripts/flamegraph.sh <path-to-frameforge-binary> [-- <binary args>]
#
# Example:
#   ./scripts/flamegraph.sh engine/build/frameforge -- \
#     --model export/temporal.onnx --input data/rendered/scene_a/seq_01/lr \
#     --output /tmp/out --threads 4
set -euo pipefail

BINARY="${1:?usage: flamegraph.sh <binary> [-- <args>]}"
shift
if [[ "${1:-}" == "--" ]]; then shift; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLAMEGRAPH_DIR="${REPO_ROOT}/.flamegraph_tools"
OUT_DIR="${REPO_ROOT}/docs"
STAMP="$(date +%Y%m%d_%H%M%S)"
PERF_DATA="/tmp/frameforge_perf_${STAMP}.data"
OUT_SVG="${OUT_DIR}/flamegraph_${STAMP}.svg"

if ! command -v perf >/dev/null 2>&1; then
  echo "error: 'perf' not found. Install it (e.g. 'sudo apt install linux-tools-common" >&2
  echo "linux-tools-\$(uname -r)') -- this script only runs on Linux with perf available." >&2
  exit 1
fi

if [[ ! -d "${FLAMEGRAPH_DIR}" ]]; then
  echo "cloning FlameGraph scripts to ${FLAMEGRAPH_DIR}"
  git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "${FLAMEGRAPH_DIR}"
fi

echo "recording: perf record -F 999 -g -o ${PERF_DATA} -- ${BINARY} $*"
perf record -F 999 -g -o "${PERF_DATA}" -- "${BINARY}" "$@"

echo "generating flamegraph -> ${OUT_SVG}"
perf script -i "${PERF_DATA}" \
  | "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" \
  | "${FLAMEGRAPH_DIR}/flamegraph.pl" --title "FrameForge engine CPU profile (${STAMP})" \
  > "${OUT_SVG}"

echo "wrote ${OUT_SVG}"
echo "commit this SVG to docs/ per CLAUDE.md 7.5, and update docs/RESULTS.md's"
echo "\"where the time goes\" section with what it shows."
