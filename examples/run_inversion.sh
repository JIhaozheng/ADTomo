#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC="${NPROC:-5}"
ITERATIONS="${ITERATIONS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
SPACING="${SPACING:-4.0}"
LAMBDA_VP="${LAMBDA_VP:-0.0}"
LAMBDA_VS="${LAMBDA_VS:-0.0}"
ALPHA_VP="${ALPHA_VP:-0.0}"
ALPHA_VS="${ALPHA_VS:-0.0}"

ARGS=(
    --iterations "$ITERATIONS"
    --learning-rate "$LEARNING_RATE"
    --spacing "$SPACING"
    --lambda-vp "$LAMBDA_VP"
    --lambda-vs "$LAMBDA_VS"
    --alpha-vp "$ALPHA_VP"
    --alpha-vs "$ALPHA_VS"
)

# Extra command-line arguments come last, so argparse treats them as overrides.
if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/inversion.py" "${ARGS[@]}" "$@"
else
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        "$SCRIPT_DIR/inversion.py" \
        "${ARGS[@]}" \
        "$@"
fi
