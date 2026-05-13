#!/usr/bin/env bash
# =============================================================================
# step5.sh
#
# Convenience launcher for Step 5 trajectory replay variants.
# Opens THREE separate terminals in sequence:
#
#   Terminal 1 (data_extraction)  : 3C_setup_carla.py             -> starts CARLA
#   Terminal 2 (data_extraction)  : 3F_generate_carla_scenario.py -> loads the map
#   Terminal 3 (env depends on --mode):
#     --mode 5A  ->  data_extraction  ->  5A_trajectory_only_carla.py
#                    (CARLA-only replay, no Gaussian Splatting)
#     --mode 5C  ->  nerfstudio       ->  5C_trajectory_replay.py
#                    (Replay with Gaussian Splatting rendering)
#
# Default mode is 5A.
#
# Usage:
#     bash step5.sh                 # defaults to 5A
#     bash step5.sh --mode 5A       # same as default
#     bash step5.sh --mode 5C       # GS replay
#     bash step5.sh -m 5C
# =============================================================================

set -e

# -------------------------- CONFIG (edit if needed) --------------------------

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Conda envs
ENV_CARLA="data_extraction"
ENV_GS="nerfstudio"

# Python scripts (paths relative to PROJECT_ROOT)
SCRIPT_3C="3_generate_simulation_data/3C_setup_carla.py"
SCRIPT_3F="3_generate_simulation_data/3F_generate_carla_scenario.py"
SCRIPT_5A="5_execute_simulation/5A_trajectory_only_carla.py"
SCRIPT_5C="5_execute_simulation/5C_trajectory_replay.py"

# CARLA RPC port
CARLA_HOST="127.0.0.1"
CARLA_PORT="2000"

# Max seconds to wait for CARLA before giving up
CARLA_WAIT_TIMEOUT=120

# Delay between Map loaded (3F) and Replay (5A/5C)
DELAY_AFTER_MAP_S=5

# -----------------------------------------------------------------------------

# Parse args
MODE="5A"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "        Use --mode 5A or --mode 5C"
            exit 1
            ;;
    esac
done

# Validate mode and pick script + env
case "$MODE" in
    5A|5a)
        REPLAY_SCRIPT="$SCRIPT_5A"
        REPLAY_ENV="$ENV_CARLA"
        REPLAY_LABEL="5A (CARLA only)"
        ;;
    5C|5c)
        REPLAY_SCRIPT="$SCRIPT_5C"
        REPLAY_ENV="$ENV_GS"
        REPLAY_LABEL="5C (Gaussian Splatting)"
        ;;
    *)
        echo "[ERROR] Invalid --mode '$MODE'. Use 5A or 5C."
        exit 1
        ;;
esac

echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] MODE         = $REPLAY_LABEL"
echo "[INFO] Replay env   = $REPLAY_ENV"
echo "[INFO] Replay script= $REPLAY_SCRIPT"

# Detect terminal emulator
if command -v gnome-terminal >/dev/null 2>&1; then
    TERM_CMD="gnome-terminal"
elif command -v xterm >/dev/null 2>&1; then
    TERM_CMD="xterm"
else
    echo "[ERROR] Neither gnome-terminal nor xterm found. Install one of them."
    exit 1
fi
echo "[INFO] Using terminal emulator: $TERM_CMD"

# Detect conda.sh
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[ERROR] Could not locate conda.sh. Edit step5.sh to set CONDA_SH manually."
    exit 1
fi
echo "[INFO] Using conda.sh: $CONDA_SH"

# Sanity check: do the python scripts exist?
for s in "$SCRIPT_3C" "$SCRIPT_3F" "$REPLAY_SCRIPT"; do
    if [ ! -f "$PROJECT_ROOT/$s" ]; then
        echo "[ERROR] Script not found: $PROJECT_ROOT/$s"
        exit 1
    fi
done

# Helper: spawn a new terminal that sources conda, activates env, runs a command.
spawn_terminal() {
    local title="$1"
    local env_name="$2"
    local command_to_run="$3"

    local full_bash_cmd="
echo '========================================================';
echo ' $title';
echo '========================================================';
source '$CONDA_SH';
conda activate '$env_name';
cd '$PROJECT_ROOT';
echo '[step5] PWD: '\$(pwd);
echo '[step5] ENV: $env_name';
echo '[step5] CMD: $command_to_run';
echo;
$command_to_run;
echo;
echo '----- Command finished. Press ENTER to close. -----';
read
"

    if [ "$TERM_CMD" = "gnome-terminal" ]; then
        gnome-terminal --title="$title" -- bash -c "$full_bash_cmd"
    else
        xterm -T "$title" -e bash -c "$full_bash_cmd" &
    fi
}

# Helper: wait until CARLA RPC port is reachable (or timeout)
wait_for_carla() {
    echo "[INFO] Waiting for CARLA RPC on ${CARLA_HOST}:${CARLA_PORT} ..."
    local elapsed=0
    while ! (echo > /dev/tcp/${CARLA_HOST}/${CARLA_PORT}) 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $CARLA_WAIT_TIMEOUT ]; then
            echo "[ERROR] Timed out waiting for CARLA on ${CARLA_HOST}:${CARLA_PORT} after ${CARLA_WAIT_TIMEOUT}s"
            return 1
        fi
        echo "  ... still waiting (${elapsed}s)"
    done
    echo "[INFO] CARLA is up (RPC port reachable)."
    return 0
}

echo ""
echo "[STEP 1/3] Spawning Terminal 1: CARLA (3C_setup_carla.py)"
spawn_terminal "Terminal 1 - CARLA" "$ENV_CARLA" "python $SCRIPT_3C"

sleep 2

if ! wait_for_carla; then
    echo "[ERROR] CARLA never became ready. Aborting."
    echo "        Check Terminal 1 for errors."
    exit 1
fi

echo ""
echo "[STEP 2/3] Spawning Terminal 2: Map (3F_generate_carla_scenario.py)"
spawn_terminal "Terminal 2 - Map" "$ENV_CARLA" "python $SCRIPT_3F"

sleep $DELAY_AFTER_MAP_S

echo ""
echo "[STEP 3/3] Spawning Terminal 3: Replay $REPLAY_LABEL"
spawn_terminal "Terminal 3 - Replay $REPLAY_LABEL" "$REPLAY_ENV" "python $REPLAY_SCRIPT"

echo ""
echo "[INFO] All three terminals launched (mode: $REPLAY_LABEL)."
echo "[INFO] Watch Terminal 1 (CARLA) and Terminal 2 (Map) for status."
echo "[INFO] Terminal 3 should now be replaying the trajectory."