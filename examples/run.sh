#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nproc="${NPROC:-1}"

if [[ "$nproc" -eq 1 ]]; then
    python "$script_dir/example.py" "$@"
else
    torchrun --standalone --nproc_per_node="$nproc" "$script_dir/example.py" "$@"
fi
