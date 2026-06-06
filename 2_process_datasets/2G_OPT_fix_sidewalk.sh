#!/bin/bash
#
# 2G_OPT_fix_sidewalk.sh
#
# Replace 'sidewalk' lane type with 'parking' in the generated OpenDRIVE map.
#
# Reads / writes (in place):
#     data/processed_dataset/<BAG>/maps/map.xodr

set -e

# ---------- Usage ----------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <bag_name.bag>"
    echo "Example: $0 reference_bag.bag"
    exit 1
fi

BAG_NAME="$1"
BAG_STEM="${BAG_NAME%.bag}"   # strip .bag extension if present

MAP_FILE="data/processed_dataset/${BAG_STEM}/maps/map.xodr"

if [ ! -f "$MAP_FILE" ]; then
    echo "ERROR: Map file not found: $MAP_FILE"
    echo "       Run step 2C first to generate the OpenDRIVE map."
    exit 1
fi

echo "Patching sidewalks -> parking in: $MAP_FILE"
sed -i 's/type="sidewalk"/type="parking"/g' "$MAP_FILE"
echo "Done."