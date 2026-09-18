#!/usr/bin/env bash
# Run the 3-D (MODEL=3d) or 1-D (MODEL=1d) synthetic inversion, serially or under torchrun.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-1d}"                 # 3d | 1d
NPROC="${NPROC:-1}"
TRAINABLE="${TRAINABLE:-vp,vs}"      # subset of vp,vs,event_loc,event_time
OPTIMIZER="${OPTIMIZER:-lbfgs}"      # lbfgs | adam
ITERATIONS="${ITERATIONS:-30}"
LAMBDA_VP="${LAMBDA_VP:-0.0}"
LAMBDA_VS="${LAMBDA_VS:-0.0}"
ALPHA_VP="${ALPHA_VP:-0.0}"
ALPHA_VS="${ALPHA_VS:-0.0}"

case "$MODEL" in
    3d) SPACING="${SPACING:-2.0}" ;;
    1d) SPACING="${SPACING:-1.0}" ;;
    *) echo "MODEL must be 3d or 1d, got '$MODEL'" >&2; exit 1 ;;
esac

ARGS=(
    --trainable "$TRAINABLE"
    --optimizer "$OPTIMIZER"
    --iterations "$ITERATIONS"
    --spacing "$SPACING"
    --lambda-vp "$LAMBDA_VP"
    --lambda-vs "$LAMBDA_VS"
    --alpha-vp "$ALPHA_VP"
    --alpha-vs "$ALPHA_VS"
)
if [[ -n "${LEARNING_RATE:-}" ]]; then
    ARGS+=(--learning-rate "$LEARNING_RATE")
fi

# Extra command-line arguments come last, so argparse treats them as overrides.
if [[ "$NPROC" -eq 1 ]]; then
    python "$SCRIPT_DIR/inversion${MODEL}.py" "${ARGS[@]}" "$@"
else
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        "$SCRIPT_DIR/inversion${MODEL}.py" \
        "${ARGS[@]}" \
        "$@"
fi
