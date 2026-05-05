from config import DEBUG_PRINT


def debug_hero_car_spawn(hero_car_point, hero_car_heading, side):
    if not DEBUG_PRINT:
        return
    print("🚗 Hero-Car-Spawn set at:", hero_car_point[0], hero_car_point[1])
    print(f"🧭 Hero-Car-Heading: {hero_car_heading:.2f}° (Side: {side})")

def debug_hero_spawn_line_error():
    if not DEBUG_PRINT:
        return
    print("❌ No hero spawn line found near click point.")

def debug_spawn_line_distance(min_dist):
    if not DEBUG_PRINT:
        return
    print("❌ Click too far from any line (distance:", round(min_dist, 5), ")")

def debug_parking_area_created(side, carla_start, carla_end, heading):
    if not DEBUG_PRINT:
        return
    print("\n📦 Parking Area created:")
    print(f"  Side: {side}")
    print(f"  Start: {carla_start}")
    print(f"  End:   {carla_end}")
    print(f"  Heading: {heading:.2f}°")

def debug_loading_osm_data(address):
    print("📡 Loading OSM data from address:", address)