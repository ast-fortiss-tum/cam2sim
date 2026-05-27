#!/usr/bin/env bash
# =============================================================================
# step5_sd.sh
#
# Convenience launcher for Step 5 (Stable Diffusion branch, 4 modes).
#
# Mode  | Script(s)                              | Env          | DAVE-2 server?
# ------+----------------------------------------+--------------+----------------
# 5A    | 5A_sd_trajectory_only_carla.py         | stable_diff  | NO
# 5B    | 5B_dave2_only_carla.py                 | stable_diff  | YES
# 5C    | 5A_sd_trajectory_only_carla.py THEN    | stable_diff  | NO
#       | 5E_stable_diff_offline_generation.py   |              |
# 5D    | 5F_sd_dave2.py                         | stable_diff  | YES
#
# All modes use 3F_sd_generate_carla_scenario.py to populate the world
# (different from step5.sh which uses 3F_generate_carla_scenario.py).
#
# Sequence:
#   1. Terminal 1: starts CARLA (3C_setup_carla.py)
#   2. Waits for CARLA RPC port to be reachable
#   3. Terminal 2: loads map + spawns cars (3F_sd_generate_carla_scenario.py)
#      Script waits for Terminal 2 to FINISH (3F_sd is fire-and-exit)
#   4. (only 5B/5D) Terminal 3: starts DAVE-2 server (communicator.py)
#      Short pause to let it bind
#   5. Terminal 4: runs the chosen Step 5 SD script (or scripts, for 5C)
#
# Usage:
#     bash step5_sd.sh                 # defaults to 5C
#     bash step5_sd.sh --mode 5A
#     bash step5_sd.sh --mode 5B
#     bash step5_sd.sh --mode 5C
#     bash step5_sd.sh -m 5D
# =============================================================================

set -e

# -------------------------- CONFIG (edit if needed) --------------------------

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Conda envs
ENV_CARLA="data_extraction"   # for 3C and 3F_sd (CARLA-side setup)
ENV_SD="stable_diff"          # for the SD-branch step 5 scripts
ENV_DAVE="dave_2"             # for the DAVE-2 TCP server

# Python scripts (paths relative to PROJECT_ROOT)
SCRIPT_3C="3_generate_simulation_data/3C_setup_carla.py"
SCRIPT_3F_SD="3_generate_simulation_data/3F_sd_generate_carla_scenario.py"
SCRIPT_5A_SD="5_execute_simulation/5A_sd_trajectory_only_carla.py"
SCRIPT_5B="5_execute_simulation/5B_dave2_only_carla.py"
SCRIPT_5E_SD="5_execute_simulation/5E_stable_diff_offline_generation.py"
SCRIPT_5F_SD="5_execute_simulation/5F_sd_dave2.py"
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

# How long to wait for 3F_sd to finish loading the map+cars (sanity timeout)
MAP_LOAD_TIMEOUT=180

# -----------------------------------------------------------------------------

# Parse args
MODE="5C"
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

# Validate mode and pick script(s) + env + dave2-server flag
NEED_DAVE_SERVER=0
IS_MODE_5C=0
case "$MODE" in
    5A|5a)
        STEP5_SCRIPT="$SCRIPT_5A_SD"
        STEP5_ENV="$ENV_SD"
        STEP5_LABEL="5A SD (CARLA-only trajectory replay with instance mapping)"
        ;;
    5B|5b)
        STEP5_SCRIPT="$SCRIPT_5B"
        STEP5_ENV="$ENV_SD"
        STEP5_LABEL="5B (CARLA-only DAVE-2 drive)"
        NEED_DAVE_SERVER=1
        ;;
    5C|5c)
        # 5C is two scripts run sequentially: 5A_sd then 5E
        STEP5_SCRIPT="$SCRIPT_5A_SD"
        STEP5_SCRIPT_2="$SCRIPT_5E_SD"
        STEP5_ENV="$ENV_SD"
        STEP5_LABEL="5C SD (CARLA replay + Stable Diffusion offline generation)"
        IS_MODE_5C=1
        ;;
    5D|5d)
        STEP5_SCRIPT="$SCRIPT_5F_SD"
        STEP5_ENV="$ENV_SD"
        STEP5_LABEL="5D SD (Stable Diffusion DAVE-2 drive)"
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
if [ $IS_MODE_5C -eq 1 ]; then
    echo "[INFO] Step 5 script (2)  = $STEP5_SCRIPT_2"
fi
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
    echo "[ERROR] Could not locate conda.sh. Edit step5_sd.sh to set CONDA_SH manually."
    exit 1
fi
echo "[INFO] conda.sh           = $CONDA_SH"

# Sanity check: do the python scripts exist?
CHECK_SCRIPTS=("$SCRIPT_3C" "$SCRIPT_3F_SD" "$STEP5_SCRIPT")
if [ $IS_MODE_5C -eq 1 ]; then
    CHECK_SCRIPTS+=("$STEP5_SCRIPT_2")
fi
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
echo '[step5_sd] PWD: '\$(pwd);
echo '[step5_sd] ENV: $env_name';
echo '[step5_sd] CMD: $command_to_run';
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
echo '[step5_sd] PWD: '\$(pwd);
echo '[step5_sd] ENV: $env_name';
echo '[step5_sd] CMD: $command_to_run';
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
SENTINEL_3F="$TMP_DIR/3F_sd_done"
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
echo "[STEP 2] Spawning Terminal 2: Map + Cars (3F_sd_generate_carla_scenario.py)"
spawn_terminal_with_sentinel \
    "Terminal 2 - Map (SD)" \
    "$ENV_CARLA" \
    "python $SCRIPT_3F_SD" \
    "$SENTINEL_3F"

if ! wait_for_sentinel "$SENTINEL_3F" "$MAP_LOAD_TIMEOUT" "3F_sd (map + cars + color map)"; then
    echo "[ERROR] 3F_sd did not finish in time. Aborting."
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

if [ $IS_MODE_5C -eq 1 ]; then
    # Mode 5C: run two scripts back-to-back in the same terminal.
    # 5A_sd captures replay frames, then 5E generates SD images offline.
    echo "[STEP $NEXT_TERM_NUM] Spawning Terminal $NEXT_TERM_NUM: $STEP5_LABEL (5A_sd + 5E)"
    spawn_terminal \
        "Terminal $NEXT_TERM_NUM - $STEP5_LABEL" \
        "$STEP5_ENV" \
        "python $STEP5_SCRIPT && python $STEP5_SCRIPT_2"
else
    echo "[STEP $NEXT_TERM_NUM] Spawning Terminal $NEXT_TERM_NUM: $STEP5_LABEL"
    spawn_terminal \
        "Terminal $NEXT_TERM_NUM - $STEP5_LABEL" \
        "$STEP5_ENV" \
        "python $STEP5_SCRIPT"
fi

echo ""
echo "[INFO] All terminals launched (mode: $STEP5_LABEL)."
echo "[INFO] Watch the terminals for status/errors."