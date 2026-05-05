def ensure_carla_functionality():
    try:
        import carla
    except ImportError:
        print(
            "⚠️ Carla is required for this script to generate .xodr files. Please install Carla or disable the Carla option with --no_carla.")
        exit(1)