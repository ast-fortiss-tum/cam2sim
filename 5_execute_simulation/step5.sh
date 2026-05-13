#!/usr/bin/env bash
# =============================================================================
# step5.sh
#
# Convenience launcher for Step 5 (4 modes).
#
# Mode  | Script                       | Env             | DAVE-2 server?
# ------+------------------------------+-----------------+----------------
# 5A    | 5A_trajectory_only_carla.py  | data_extraction | NO
# 5B    | 5B_dave2_only_carla.py       | data_extraction | YES
# 5C    | 5C_trajectory_replay.py      | nerfstudio      | NO
# 5D    | 5D_dave2.py                  | nerfstudio      | YES
#
# Sequence:
#   1. Terminal 1: starts CARLA (3C_setup_carla.py)
#   2. Waits for CARLA RPC port to be reachable
#   3. Terminal 2: loads map + spawns cars (3F_generate_carla_scenario.py)
#      Script waits for Terminal 2 to FINISH (3F is fire-and-exit)
#   4. (only 5B/5D) Terminal 3: starts DAVE-2 server (communicator.py)
#      Short pause to let it bind
#   5. Terminal 4: runs the chosen Step 5 script
#
# Usage:
#     bash step5.sh                 # defaults to 5A
#     bash step5.sh --mode 5A
#     bash step5.sh --mode 5B
#     bash step5.sh --mode 5C
#     bash step5.sh -m 5D
# =============================================================================

set -e

# -------------------------- CONFIG (edit if needed) --------------------------

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Conda envs
ENV_CARLA="data_extraction"
ENV_GS="nerfstudio"
ENV_DAVE="dave_2"

# Python scripts (paths relative to PROJECT_ROOT)
SCRIPT_3C="3_generate_simulation_data/3C_setup_carla.py"
SCRIPT_3F="3_generate_simulation_data/3F_generate_carla_scenario.py"
SCRIPT_5A="5_execute_simulation/5A_trajectory_only_carla.py"
SCRIPT_5B="5_execute_simulation/5B_dave2_only_carla.py"
SCRIPT_5C="5_execute_simulation/5C_trajectory_replay.py"
SCRIPT_5D="5_execute_simulation/5D_dave2.py"
SCRIPT_DAVE_SERVER="system_under_test/communicator.py"

# CARLA RPC port (only used to wait for CARLA to be ready)
CARLA_HOST="127.0.0.1"
CARLA_PORT="2000"
CARLA_WAIT_TIMEOUT=120

# DAVE-2 socket server (communicator.py)
# Hardcoded inside communicator.py: HOST=localhost, PORT=5090
DAVE_HOST="127.0.0.1"
DAVE_PORT="5090"
DAVE_WAIT_TIMEOUT=120

# Subfolder where communicator.py lives (and where final.h5 sits next to it)
DAVE_SERVER_CWD="system_under_test"

# How long to wait for 3F to finish loading the map+cars (sanity timeout)
MAP_LOAD_TIMEOUT=180

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
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "        Use --mode 5A | 5B | 5C | 5D"
            exit 1
            ;;
    esac
done

# Validate mode and pick script + env + dave2-server flag
NEED_DAVE_SERVER=0
case "$MODE" in
    5A|5a)
        STEP5_SCRIPT="$SCRIPT_5A"
        STEP5_ENV="$ENV_CARLA"
        STEP5_LABEL="5A (CARLA-only trajectory replay)"
        ;;
    5B|5b)
        STEP5_SCRIPT="$SCRIPT_5B"
        STEP5_ENV="$ENV_CARLA"
        STEP5_LABEL="5B (CARLA-only DAVE-2 drive)"
        NEED_DAVE_SERVER=1
        ;;
    5C|5c)
        STEP5_SCRIPT="$SCRIPT_5C"
        STEP5_ENV="$ENV_GS"
        STEP5_LABEL="5C (Gaussian Splatting trajectory replay)"
        ;;
    5D|5d)
        STEP5_SCRIPT="$SCRIPT_5D"
        STEP5_ENV="$ENV_GS"
        STEP5_LABEL="5D (Gaussian Splatting DAVE-2 drive)"
        NEED_DAVE_SERVER=1
        ;;
    *)
        echo "[ERROR] Invalid --mode '$MODE'. Use 5A | 5B | 5C | 5D."
        exit 1
        ;;
esac

echo "[INFO] PROJECT_ROOT       = $PROJECT_ROOT"
echo "[INFO] MODE               = $STEP5_LABEL"
echo "[INFO] Step 5 env         = $STEP5_ENV"
echo "[INFO] Step 5 script      = $STEP5_SCRIPT"
echo "[INFO] DAVE-2 server      = $( [ $NEED_DAVE_SERVER -eq 1 ] && echo YES || echo NO )"

# Detect terminal emulator
if command -v gnome-terminal >/dev/null 2>&1; then
    TERM_CMD="gnome-terminal"
elif command -v xterm >/dev/null 2>&1; then
    TERM_CMD="xterm"
else
    echo "[ERROR] Neither gnome-terminal nor xterm found. Install one of them."
    exit 1
fi
echo "[INFO] Terminal emulator  = $TERM_CMD"

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
echo "[INFO] conda.sh           = $CONDA_SH"

# Sanity check: do the python scripts exist?
CHECK_SCRIPTS=("$SCRIPT_3C" "$SCRIPT_3F" "$STEP5_SCRIPT")
if [ $NEED_DAVE_SERVER -eq 1 ]; then
    CHECK_SCRIPTS+=("$SCRIPT_DAVE_SERVER")
fi
for s in "${CHECK_SCRIPTS[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$s" ]; then
        echo "[ERROR] Script not found: $PROJECT_ROOT/$s"
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# spawn_terminal TITLE ENV "command"
# Opens a new terminal, sources conda, activates env, runs command,
# keeps window open after exit.
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

# spawn_terminal_with_sentinel TITLE ENV "command" SENTINEL_FILE
# Same as spawn_terminal but touches a sentinel file when the command exits.
# Useful to wait for the subprocess to finish from the parent script.
spawn_terminal_with_sentinel() {
    local title="$1"
    local env_name="$2"
    local command_to_run="$3"
    local sentinel="$4"

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
exit_code=\$?;
echo;
echo \"----- Command finished (exit \$exit_code). Press ENTER to close. -----\";
touch '$sentinel';
read
"

    if [ "$TERM_CMD" = "gnome-terminal" ]; then
        gnome-terminal --title="$title" -- bash -c "$full_bash_cmd"
    else
        xterm -T "$title" -e bash -c "$full_bash_cmd" &
    fi
}

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

wait_for_dave2() {
    echo "[INFO] Waiting for DAVE-2 server on ${DAVE_HOST}:${DAVE_PORT} ..."
    local elapsed=0
    while ! (echo > /dev/tcp/${DAVE_HOST}/${DAVE_PORT}) 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $DAVE_WAIT_TIMEOUT ]; then
            echo "[ERROR] Timed out waiting for DAVE-2 server on ${DAVE_HOST}:${DAVE_PORT} after ${DAVE_WAIT_TIMEOUT}s"
            echo "        (Loading the TF model + 'final.h5' takes a while; if your hardware is slow,"
            echo "         increase DAVE_WAIT_TIMEOUT at the top of this script.)"
            return 1
        fi
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "  ... still waiting (${elapsed}s) — TF model is probably still loading"
        fi
    done
    echo "[INFO] DAVE-2 server is up (port reachable)."
    return 0
}

# Wait for a sentinel file to appear, up to TIMEOUT seconds.
wait_for_sentinel() {
    local sentinel="$1"
    local timeout="$2"
    local label="$3"

    echo "[INFO] Waiting for $label to finish (sentinel: $sentinel) ..."
    local elapsed=0
    while [ ! -f "$sentinel" ]; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $timeout ]; then
            echo "[ERROR] Timed out waiting for $label after ${timeout}s"
            return 1
        fi
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "  ... still waiting (${elapsed}s)"
        fi
    done
    echo "[INFO] $label finished."
    return 0
}

# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

TMP_DIR="$(mktemp -d)"
SENTINEL_3F="$TMP_DIR/3F_done"
trap "rm -rf '$TMP_DIR'" EXIT

echo ""
echo "[STEP 1] Spawning Terminal 1: CARLA (3C_setup_carla.py)"
spawn_terminal "Terminal 1 - CARLA" "$ENV_CARLA" "python $SCRIPT_3C"

sleep 2

if ! wait_for_carla; then
    echo "[ERROR] CARLA never became ready. Aborting."
    echo "        Check Terminal 1 for errors."
    exit 1
fi

echo ""
echo "[STEP 2] Spawning Terminal 2: Map + Cars (3F_generate_carla_scenario.py)"
spawn_terminal_with_sentinel \
    "Terminal 2 - Map" \
    "$ENV_CARLA" \
    "python $SCRIPT_3F" \
    "$SENTINEL_3F"

if ! wait_for_sentinel "$SENTINEL_3F" "$MAP_LOAD_TIMEOUT" "3F (map + cars)"; then
    echo "[ERROR] 3F did not finish in time. Aborting."
    exit 1
fi

# Optional: short cushion after map load
sleep 2

if [ $NEED_DAVE_SERVER -eq 1 ]; then
    echo ""
    echo "[STEP 3] Spawning Terminal 3: DAVE-2 server ($SCRIPT_DAVE_SERVER)"
    # communicator.py loads 'final.h5' from its current working directory,
    # so we cd into system_under_test/ before launching it.
    spawn_terminal \
        "Terminal 3 - DAVE-2 server" \
        "$ENV_DAVE" \
        "cd $DAVE_SERVER_CWD && python $(basename $SCRIPT_DAVE_SERVER)"

    if ! wait_for_dave2; then
        echo "[ERROR] DAVE-2 server never became ready. Aborting."
        echo "        Check Terminal 3 for TensorFlow/h5 errors."
        exit 1
    fi
fi

NEXT_TERM_NUM=$(( NEED_DAVE_SERVER == 1 ? 4 : 3 ))
echo ""
echo "[STEP $NEXT_TERM_NUM] Spawning Terminal $NEXT_TERM_NUM: $STEP5_LABEL"
spawn_terminal \
    "Terminal $NEXT_TERM_NUM - $STEP5_LABEL" \
    "$STEP5_ENV" \
    "python $STEP5_SCRIPT"

echo ""
echo "[INFO] All terminals launched (mode: $STEP5_LABEL)."
echo "[INFO] Watch the terminals for status/errors."