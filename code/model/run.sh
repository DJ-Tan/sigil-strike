#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  SIGIL STRIKE model dispatcher
#
# Usage:
#   ./run.sh <model> <action> [--team N] [extra args...]
#
#   model   : landmark | cnn | auto
#   action  : collect | train | infer
#
#   "auto" reads MODEL_TYPE from Teams/TeamN/team.env (requires --team N)
#
# Examples:
#   ./run.sh landmark collect --team 1
#   ./run.sh landmark train   --team 1 --model rf
#   ./run.sh landmark infer   --team 1 --player 1
#
#   ./run.sh cnn      collect --team 2
#   ./run.sh cnn      train   --team 2 --epochs 20
#   ./run.sh cnn      infer   --team 2 --player 2 --cam 1
#
#   ./run.sh auto     infer   --team 3 --player 1   # reads MODEL_TYPE from team.env
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"
ACTION="${2:-}"

usage() {
    echo ""
    echo "  Usage: $0 <model> <action> [--team N] [args...]"
    echo ""
    echo "  model   : landmark | cnn | auto"
    echo "  action  : collect | train | infer"
    echo ""
    echo "  Examples:"
    echo "    $0 landmark collect --team 1"
    echo "    $0 landmark train   --team 1"
    echo "    $0 landmark infer   --team 1 --player 1"
    echo "    $0 cnn      collect --team 2"
    echo "    $0 cnn      train   --team 2 --epochs 20"
    echo "    $0 cnn      infer   --team 2 --player 2"
    echo "    $0 auto     infer   --team 3 --player 1"
    echo ""
    exit 1
}

if [[ -z "$MODEL" || -z "$ACTION" ]]; then
    usage
fi

shift 2   # remaining args passed through to the Python script

# ── Resolve "auto" model from team.env ───────────────────────────────────────
if [[ "$MODEL" == "auto" ]]; then
    TEAM_NUM=""
    for arg in "$@"; do
        if [[ "$TEAM_NUM" == "next" ]]; then
            TEAM_NUM="$arg"
            break
        fi
        [[ "$arg" == "--team" ]] && TEAM_NUM="next"
    done

    if [[ -z "$TEAM_NUM" || "$TEAM_NUM" == "next" ]]; then
        echo "[run.sh] 'auto' requires --team N to be specified."
        usage
    fi

    ENV_FILE="$PROJECT_ROOT/Teams/Team${TEAM_NUM}/team.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "[run.sh] team.env not found: $ENV_FILE"
        exit 1
    fi

    MODEL=$(grep -E "^\s*MODEL_TYPE\s*=" "$ENV_FILE" \
            | tail -1 | sed 's/.*=\s*//' | tr -d '[:space:]')

    if [[ -z "$MODEL" ]]; then
        echo "[run.sh] MODEL_TYPE not set in $ENV_FILE — defaulting to 'landmark'."
        MODEL="landmark"
    fi

    echo "[run.sh] Team ${TEAM_NUM} MODEL_TYPE=${MODEL}"
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$MODEL" in
    landmark)
        case "$ACTION" in
            collect) python "$SCRIPT_DIR/landmark/collect_data.py" "$@" ;;
            train)   python "$SCRIPT_DIR/landmark/train_model.py"  "$@" ;;
            infer)   python "$SCRIPT_DIR/landmark/inference.py"    "$@" ;;
            *) echo "[run.sh] Unknown action '$ACTION'"; usage ;;
        esac
        ;;
    cnn)
        case "$ACTION" in
            collect) python "$SCRIPT_DIR/cnn/collect_data.py"  "$@" ;;
            train)   python "$SCRIPT_DIR/cnn/train_model.py"   "$@" ;;
            infer)   python "$SCRIPT_DIR/cnn/inference.py"     "$@" ;;
            *) echo "[run.sh] Unknown action '$ACTION'"; usage ;;
        esac
        ;;
    *)
        echo "[run.sh] Unknown model '$MODEL' — expected: landmark | cnn | auto"
        usage
        ;;
esac
