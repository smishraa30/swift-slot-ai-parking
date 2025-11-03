# config.py
MODEL_PATH = 'best.pt'
VIDEO_SOURCES = {
    "North_Lot": "carPark1.mp4",
    "South_Garage": "carPark2.mp4",
    "East_Field": "carPark.mp4",
}
ENTRY_POINTS = {
    "North_Lot": (320, 480),
    "South_Garage": (100, 450),
    "East_Field": (600, 450),
}

# --- Navigation Angle Constants ---
ANGLE_FORWARD = (75, 105)
ANGLE_FORWARD_RIGHT = (45, 75)
# ... etc.