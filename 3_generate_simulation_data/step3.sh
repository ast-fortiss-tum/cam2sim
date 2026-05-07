SCRIPTS=(
    "3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
    "3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py"
    "3_generate_simulation_data/3C_setup_carla.py"
    "3_generate_simulation_data/3F_generate_carla_scenario.py"
)

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

        sleep 5
    fi
done

echo ""
echo "All Scripts for Step 3 completed successfully"