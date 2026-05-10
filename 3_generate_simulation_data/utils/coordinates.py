import math
import os
import numpy as np
from pyproj import Transformer, CRS

# ==========================================
# ⚙️ GEOGRAPHIC CONSTANTS (From actual first pose in trajectory.txt)
# ==========================================
LAT0 = 48.17559486315574
LON0 = 11.5951469668968
ODOM0_X = 692932.695045   # First trajectory point X (UTM Easting)
ODOM0_Y = 5339067.061488  # First trajectory point Y (UTM Northing)
YAW0 = 1.058725           # First trajectory point Yaw (radians) - vehicle heading

# 692932.695045,5339067.061488,549.924500,1.058725

# ==========================================
# ⚙️ COORDINATE TRANSFORMERS
# ==========================================
# UTM Zone 32N (EPSG:25832) - covers Munich area, IF U WANT TO USE DATA IN DIFFERENT AREAS NEED TO CHANGE FROM EPSG!!!
transformer_utm_to_wgs84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def utm_to_wgs84(easting, northing):
    """
    Convert UTM Zone 32N coordinates to WGS84 lat/lon.
    This is the proper way to convert - no approximation needed.
    """
    lon, lat = transformer_utm_to_wgs84.transform(easting, northing)
    return lat, lon

def enu_to_latlon_precise(dx_m, dy_m):
    """
    Conversione precisa ENU -> WGS84 (legacy function for compatibility).
    """
    lat_ref_rad = math.radians(LAT0)

    m_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat_ref_rad)
        + 1.175 * math.cos(4 * lat_ref_rad)
    )

    m_per_deg_lon = (
        111412.84 * math.cos(lat_ref_rad)
        - 93.5 * math.cos(3 * lat_ref_rad)
    )

    return LAT0 + (dy_m / m_per_deg_lat), LON0 + (dx_m / m_per_deg_lon)

# ------------------------------------------------------------------
# MAIN CONVERSION FUNCTION
# ------------------------------------------------------------------
def odom_xy_to_wgs84_vec(x_arr, y_arr):
    """
    Convert odometry coordinates (UTM Zone 32N) to WGS84 lat/lon.
    
    Parameters:
        x_arr: UTM Easting values (meters)
        y_arr: UTM Northing values (meters)
    
    Returns:
        lat, lon arrays
    """
    if len(x_arr) == 0:
        return np.array([]), np.array([])

    x_arr = np.asarray(x_arr)
    y_arr = np.asarray(y_arr)

    # Convert UTM to WGS84 using pyproj (exact transformation)
    lon, lat = transformer_utm_to_wgs84.transform(x_arr, y_arr)

    return np.asarray(lat), np.asarray(lon)

def get_projected_coords(x_arr, y_arr):
    """
    Wrapper per il Tool di Visualizzazione.
    Chiama la funzione sopra e poi proietta per la mappa web.
    """
    # Chiama la logica comune passando i parametri
    lat, lon = odom_xy_to_wgs84_vec(x_arr, y_arr)
    
    if len(lat) == 0:
        return np.array([]), np.array([])

    # Proietta per Contextily
    xm, ym = transformer_to_3857.transform(lon, lat)
    return xm, ym

def load_shift_values(centroid_file_path):
    """
    Looks for a 'shift.txt' file in the same folder as the centroid file.
    Returns dictionary with shifts or defaults (0.0).
    """
    dataset_folder = os.path.dirname(centroid_file_path)
    shift_file = os.path.join(dataset_folder, "shift.txt")
    
    defaults = {"SHIFT_X": 0.0, "SHIFT_Y": 0.0, "YAW_OFFSET": 0.0}

    if not os.path.exists(shift_file):
        print(f"⚠️  Warning: 'shift.txt' not found in {dataset_folder}. Using defaults (0.0).")
        return defaults

    print(f"Loading calibration from: {shift_file}")
    vals = defaults.copy()
    try:
        with open(shift_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=")
                    if key in vals:
                        vals[key] = float(val)
        return vals
    except Exception as e:
        print(f"Error reading shift.txt: {e}. Using defaults.")
        return defaults