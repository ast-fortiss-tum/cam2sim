import matplotlib
import matplotlib.pyplot as plt
from shapely import LineString
from shapely.geometry import Point
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from utils.config import CLICK_DISTANCE_THRESHOLD, HERO_CAR_OFFSET_METERS, SPAWN_OFFSET_METERS, SPAWN_OFFSET_METERS_LEFT, SPAWN_OFFSET_METERS_RIGHT
from utils.debug import debug_hero_car_spawn, debug_hero_spawn_line_error, debug_spawn_line_distance, \
    debug_parking_area_created
from utils.map_data import generate_spawn_gdf, get_heading, latlon_to_carla, get_origin_lat_lon

hero_car_point, hero_car_heading, hero_car_display = None, None, None
selected_segments = []
click_points = []
all_spawns = []

matplotlib.use("TkAgg")

def close_event(event):
    plt.close()

def update_plot():
    ax.clear()
    
    # 1. Plot the base layers
    buildings.plot(ax=ax, color="lightgray")
    edges.plot(ax=ax, color="black", linewidth=1)
    
    if not spawn_gdf.empty:
        spawn_gdf.plot(ax=ax, color='red', linewidth=2)

    # 2. PLOT VEHICLES HERE (Outside the loop)
    # Note: I swapped x and y because 'x_vehicles' contains Latitude (48.x) 
    # and 'y_vehicles' contains Longitude (11.x). 
    # Matplotlib/GeoPandas expects (Lon, Lat).
    #ax.scatter(y_vheicles, x_vehicles, color="green", zorder=5, label="Parked Vehicles")

    # 3. Handle Hero Car and Titles
    if hero_car_point is not None:
        ax.plot(hero_car_display.x, hero_car_display.y, 'o', color='blue', markersize=8, label="Hero-Car")
        fig.suptitle("Select Parking-Areas by clicking on the red parking areas.")
    else:
        fig.suptitle("Select a spawn point for the hero-car by clicking next to a street in the travel direction")
    
    # 4. Highlight Selected Segments
    for seg in selected_segments:
        ax.plot(*zip(*seg.coords), color="green", linewidth=4)
        # REMOVED: ax.scatter(...) was here. It caused duplication and dependency on selection.

    # 5. Final Formatting
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    legend_handles = [
        mpatches.Patch(color="lightgray", label="Buildings"),
        mlines.Line2D([], [], color="black", linewidth=2, label="Streets"),
        mlines.Line2D([], [], color="red", linewidth=2, label="Possible Parking Areas"),
        mlines.Line2D([], [], color="green", linewidth=4, label="Selected Parking Areas"),
        mlines.Line2D([], [], color="green", marker='o', linestyle='', label="Parked Vehicles"), # Added to legend
        mlines.Line2D([], [], color="blue", marker='o', linestyle='', markersize=8, label="Spawn Point"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.canvas.draw()

def create_plot(buildings_data, edges_data, address):
    global buildings, edges, hero_spawn_gdf, spawn_gdf, fig, ax, origin_lat, origin_lon
    buildings = buildings_data
    edges = edges_data

    hero_spawn_gdf = generate_spawn_gdf(edges, offset=HERO_CAR_OFFSET_METERS, offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT, override=True)
    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT,)

    print("spawn_gdf length:", len(spawn_gdf))
    print("spawn_gdf geometry head:", spawn_gdf.geometry.head())

    fig, ax = plt.subplots(figsize=(12, 12))
    origin_lat, origin_lon = get_origin_lat_lon(edges_data, address)
    fig.canvas.manager.set_window_title("Vehicle Spawn Position Selector")

    #button_ax = fig.add_axes([0.80, 0.01, 0.18, 0.06])  # [links, unten, Breite, Höhe]
    #close_button = Button(button_ax, "Close and Save", color="lightgray", hovercolor="gray")
    #close_button.on_clicked(close_event)

    update_plot()

def show_plot():
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.axis("off")
    plt.show(block=True)

def on_click(event):
    global hero_car_point, hero_car_heading, hero_car_display

    if event.inaxes != ax:
        return

    click_pt = Point(event.xdata, event.ydata)

    if hero_car_point is None:
        nearest_line, side, _ = find_nearest_hero_spawn_line(click_pt)
        if nearest_line is None:
            debug_hero_spawn_line_error()
            return
        proj_dist = nearest_line.project(click_pt)
        proj_point = nearest_line.interpolate(proj_dist)
        hero_car_display = proj_point
        hero_car_point = latlon_to_carla(origin_lat, origin_lon, proj_point.y, proj_point.x)
        coords = list(nearest_line.coords)
        start_lat, start_lon = coords[0][1], coords[0][0]
        end_lat, end_lon = coords[-1][1], coords[-1][0]
        heading = get_heading(start_lat, start_lon, end_lat, end_lon)
        if side == "left":
            heading = (heading + 180) % 360
        hero_car_heading = heading
        update_plot()
        debug_hero_car_spawn(hero_car_point, hero_car_heading, side)
        return

    click_points.append(click_pt)

    if len(click_points) == 2:
        p1, p2 = click_points
        nearest_line, side, street_id = find_nearest_spawn_line(p1)

        if nearest_line is None:
            click_points.clear()
            return

        segment = extract_line_segment(nearest_line, p1, p2)
        selected_segments.append(segment)

        start_lat, start_lon = segment.coords[0][1], segment.coords[0][0]
        end_lat, end_lon = segment.coords[-1][1], segment.coords[-1][0]

        heading = get_heading(start_lat, start_lon, end_lat, end_lon)

        if side == "left":
            heading = (heading + 180) % 360

        if event.button != 1:
            heading = (heading + 90) % 360

        carla_start = latlon_to_carla(origin_lat, origin_lon, start_lat, start_lon)
        carla_end = latlon_to_carla(origin_lat, origin_lon, end_lat, end_lon)

        all_spawns.append({
            "side": side,
            "street_id": street_id,
            "mode": "parallel" if event.button == 1 else "perpendicular",
            "start": carla_start,
            "end": carla_end,
            "heading": heading
        })
        update_plot()
        click_points.clear()
        debug_parking_area_created(side, carla_start, carla_end, heading)


def find_nearest_spawn_line(click_pt):
    return find_nearest_line(click_pt, spawn_gdf)

def find_nearest_hero_spawn_line(click_pt):
    return find_nearest_line(click_pt, hero_spawn_gdf)

def find_nearest_line(click_pt, gdf):
    min_dist = float('inf')
    nearest_geom = None
    side = None
    for idx, geom in enumerate(gdf.geometry):
        dist = geom.distance(click_pt)
        if dist < min_dist:
            min_dist = dist
            nearest_geom = geom
            side = gdf.iloc[idx]['side']
            street_id = None
    if min_dist < CLICK_DISTANCE_THRESHOLD:
        return nearest_geom, side, street_id
    else:
        debug_spawn_line_distance(min_dist)
        return None, None, None

def substring_line(line: LineString, start_dist: float, end_dist: float) -> LineString:
    if start_dist > end_dist:
        start_dist, end_dist = end_dist, start_dist

    coords = list(line.coords)
    result = []

    dist_travelled = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        seg = LineString([p1, p2])
        seg_length = seg.length

        if dist_travelled + seg_length < start_dist:
            dist_travelled += seg_length
            continue
        if dist_travelled > end_dist:
            break

        seg_start = max(start_dist, dist_travelled)
        seg_end = min(end_dist, dist_travelled + seg_length)

        ratio_start = (seg_start - dist_travelled) / seg_length
        ratio_end = (seg_end - dist_travelled) / seg_length

        interp_start = seg.interpolate(ratio_start, normalized=True)
        interp_end = seg.interpolate(ratio_end, normalized=True)

        result.append(interp_start.coords[0])
        result.append(interp_end.coords[0])

        dist_travelled += seg_length

    return LineString(result)

def extract_line_segment(line, p1, p2):
    d1 = line.project(p1)
    d2 = line.project(p2)
    return substring_line(line, d1, d2)

def get_output(dist):
    # PRIMA ERA: if hero_car_point is None or not all_spawns:
    
    # MODIFICA: Controlliamo solo se manca la Hero Car. 
    # I parcheggi (all_spawns) possono essere vuoti (lista []).
    if hero_car_point is None:
        return None

    return {
        "offset": {
            "x": None,
            "y": None,
            "heading": None
        },
        "dist": dist,
        "hero_car": {"position": hero_car_point,
                     "heading": hero_car_heading} if hero_car_point else None,
        "spawn_positions": all_spawns, # Se è vuoto, salva [] nel JSON, che va benissimo
    }