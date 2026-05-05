import json
import os
import sys

# =======================
# PATH SETUP (workaround per import da root del progetto)
# =======================
# Aggiunge la cartella padre (root del progetto) a sys.path, cosi' funzionano
# gli import di `config` e `utils.*` indipendentemente da dove viene lanciato
# lo script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =======================
# IMPORTS
# =======================
from config import MAPS_FOLDER_NAME
# 1. ARGUMENT PARSER
from utils.argparser import parse_map_args
# 2. MAP & GEOMETRY FUNCTIONS
from utils.map_data import get_street_data, fetch_osm_data
# 3. SAVE & LOAD FUNCTIONS
from utils.save_data import (
    get_map_folder_name, 
    create_map_folders, 
    save_vehicle_data, 
    save_map_data, 
    save_osm_data, 
    get_map_data,           
    get_existing_osm_data   
)
from utils.other import ensure_carla_functionality
from utils.plotting import create_plot, show_plot, get_output
# =======================
# MAIN SCRIPT
# =======================
def main():
    # 1. PARSE ARGUMENTS
    args = parse_map_args()
    # 2. SETUP PATHS
    map_name = args.name if args.name else get_map_folder_name(args.address)
    folder_name = os.path.join(MAPS_FOLDER_NAME, map_name)
    # 3. LOAD EXISTING DATA (To preserve offsets if reloading)
    map_data = get_map_data(map_name, None, args.no_carla)
    if not args.no_carla:
        ensure_carla_functionality()
    # 4. FETCH OSM DATA (Download or Load from Disk)
    if not args.skip_fetch:
        print(f"🌍 Fetching OSM data for: {args.address}...")
        osm_data = get_street_data(args.address, dist=args.dist)
    else:
        print(f"📂 Loading existing OSM data from: {folder_name}")
        osm_data = get_existing_osm_data(folder_name)
    create_map_folders(folder_name)
    if not args.skip_fetch:
        save_osm_data(folder_name, osm_data)
    # 5. PROCESS MAP GEOMETRY
    print("🏗️  Processing map geometry (Nodes, Edges, Buildings)...")
    G, edges, buildings = fetch_osm_data(folder_name)
    # ========================
    # LOGIC BRANCHING
    # ========================
    
    # In ENTRAMBI i casi apriamo la GUI, perché ci serve SEMPRE la Hero Car.
    # Cambia solo il messaggio all'utente.
    
    print(f"\n🎨 Mode: {args.mode.upper()}. Opening GUI...")
    
    if args.mode == "manual":
        print("ℹ️  INSTRUCTIONS: Select Hero Car (Blue) AND Parking Areas (Green).")
    else:
        print("ℹ️  INSTRUCTIONS: Select ONLY the Hero Car (Blue Point).")
        print("   (Parking areas will be injected later from Clusters.txt, so you can ignore red lines).")
        print("👉 Click the start position and then CLOSE the window.")
    # Open the Plotting Window
    create_plot(buildings, edges, args.address)
    show_plot() # Code blocks here until window is closed
    # Retrieve data from the plot
    output_json = get_output(args.dist)
    # Logic to preserve previous manual offsets if re-running
    if map_data is not None and map_data.get("vehicle_data"):
        prev_data = map_data["vehicle_data"]
        if prev_data.get("dist") == args.dist and prev_data.get("offset") is not None:
            if output_json is not None:
                output_json["offset"] = prev_data["offset"]
                print("⚠️  Copied Offset Values from existing Map-Data.")
    if output_json is not None:
        # Salva SEMPRE tutto (Map + Vehicle Data con Hero Car)
        save_map_data(folder_name, osm_data, args.no_carla)
        save_vehicle_data(folder_name, output_json)
        
        print(f"\n✅ [{args.mode.upper()}] Map and Vehicle Data saved to: {folder_name}")
        
        if args.mode != "manual":
            print("💡 JSON created with Hero Car. Now run 'create_vehicle_data_from_centroids.py' to add clusters.")
    else:
        print("\n⚠️  Plot closed without selection. Nothing saved.")
        # Se l'utente chiude senza cliccare nulla, salviamo almeno la mappa geometrica per sicurezza?
        # Di solito no, perché manca il vehicle_data.json essenziale.
if __name__ == "__main__":
    main()