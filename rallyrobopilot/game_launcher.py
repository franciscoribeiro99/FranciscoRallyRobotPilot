from rallyrobopilot import Car, Track, SunLight, MultiRaySensor
from ursina import *
from pathlib import Path

def load_asset(asset_path):
    """Helper function to check if an asset exists."""
    if not Path(asset_path).exists():
        print(f"Asset not found: {asset_path}")
    else:
        print(f"Loading asset: {asset_path}")

def prepare_game_app():
    from ursina import window, Ursina

    # Create Window
    window.vsync = True  # Set to false to uncap FPS limit of 60
    app = Ursina(size=(640, 512))

    # Set assets folder. Here assets are relative to the root of the project
    # When running from scripts/main.py, assets are one folder up from the 'scripts' folder.
    application.asset_folder = Path(__file__).resolve().parent.parent / 'assets'
    print("Asset folder:", application.asset_folder)

    # Set up the window title and properties
    window.title = "Rally"
    window.borderless = False
    window.show_ursina_splash = False
    window.cog_button.disable()
    window.fps_counter.enable()
    window.exit_button.disable()

    # Define Global models & textures (ensure these are relative to the 'assets/' folder)
    global_models = [
        "cars/sports-car.obj",
        "particles/particles.obj",
        "utils/line.obj"
    ]
    global_texs = [
        "cars/garage/sports-car/sports-red.png",
        "cars/garage/sports-car/sports-blue.png",
        "cars/garage/sports-car/sports-green.png",
        "cars/garage/sports-car/sports-orange.png",
        "cars/garage/sports-car/sports-white.png",
        "particles/particle_forest_track.png",
        "utils/red.png"
    ]

    # Load and check assets before proceeding
    for model in global_models:
        load_asset(application.asset_folder / model)
    for tex in global_texs:
        load_asset(application.asset_folder / tex)

    # Load track and car models
    track_name = "VisualTrack"
    track = Track(track_name)
    print("loading assets after track creation")
    track.load_assets(global_models, global_texs)

    # Car Setup
    car = Car()
    car.sports_car()

    # Set up track for the car
    car.set_track(track)
    car.multiray_sensor = MultiRaySensor(car, 15, 90)
    car.multiray_sensor.disable()

    # Lighting + shadows
    sun = SunLight(direction=(-0.7, -0.9, 0.5), resolution=3072, car=car)
    ambient = AmbientLight(color=Vec4(0.5, 0.55, 0.66, 0) * 0.75)

    render.setShaderAuto()

    # Sky
    Sky(texture="sky")

    # Make the car visible
    car.visible = True
    mouse.locked = False
    mouse.visible = True

    # Enable car and camera settings
    car.enable()
    car.camera_angle = "top"
    car.change_camera = True
    car.camera_follow = True

    # Activate the track and start playing
    track.activate()
    track.played = True

    # Add debugging to confirm everything is loaded
    print("Game app and car successfully prepared.")
    return app, car
