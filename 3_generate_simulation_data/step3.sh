SCRIPTS=(
    "3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
    "3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py"
    "3_generate_simulation_data/3C_setup_carla.py"
    "3_generate_simulation_data/3F_generate_carla_scenario.py"
)

# CARLA RPC settings
CARLA_HOST="127.0.0.1"
CARLA_PORT="2000"
CARLA_WAIT_TIMEOUT=120

wait_for_carla() {
    echo "Waiting for CARLA RPC on ${CARLA_HOST}:${CARLA_PORT} ..."
    local elapsed=0
    while ! (echo > /dev/tcp/${CARLA_HOST}/${CARLA_PORT}) 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $CARLA_WAIT_TIMEOUT ]; then
            echo "Timed out waiting for CARLA after ${CARLA_WAIT_TIMEOUT}s"
            return 1
        fi
        echo "  ... still waiting (${elapsed}s)"
    done
    echo "CARLA is up (RPC port reachable)."
    # Small cushion to let CARLA finish initialization after port opens
    sleep 3
    return 0
}

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running Script $SCRIPT"
    if [ "$SCRIPT" != "3_generate_simulation_data/3C_setup_carla.py" ]; then
        python3 "$SCRIPT"
        if [ $? -ne 0 ]; then
            echo "Error in $SCRIPT. Aborting."
            exit 1
        fi
    else
        python3 "$SCRIPT" &
        PID=$!
        if ! wait_for_carla; then
            echo "CARLA never became ready. Aborting."
            kill $PID 2>/dev/null
            exit 1
        fi
    fi
done

echo ""
echo "All Scripts for Step 3 completed successfully"