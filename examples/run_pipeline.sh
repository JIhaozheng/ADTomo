#!/usr/bin/env bash
# Full synthetic pipeline: 00 velocity -> 01 stations -> 02 events -> 03 picks -> inversion.
# Example: MODEL=1d LOCATION_NOISE_KM=2 TIME_NOISE_S=0.5 TRAINABLE=event_loc,event_time bash run_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-3d}"
NUM_STATIONS="${NUM_STATIONS:-50}"
NUM_EVENTS="${NUM_EVENTS:-500}"
LOCATION_NOISE_KM="${LOCATION_NOISE_KM:-0.0}"
TIME_NOISE_S="${TIME_NOISE_S:-0.0}"
PICK_SPACING="${PICK_SPACING:-2.0}"

python "$SCRIPT_DIR/00_gen_velocity.py" --model "$MODEL"
python "$SCRIPT_DIR/01_gen_stations.py" --num-stations "$NUM_STATIONS"
python "$SCRIPT_DIR/02_gen_events.py" --num-events "$NUM_EVENTS" --location-noise-km "$LOCATION_NOISE_KM" --time-noise-s "$TIME_NOISE_S"
python "$SCRIPT_DIR/03_gen_picks.py" --spacing "$PICK_SPACING"
MODEL="$MODEL" bash "$SCRIPT_DIR/run_inversion.sh" "$@"
