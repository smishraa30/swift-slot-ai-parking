import cv2
import os
import math
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import threading
import config

app = Flask(__name__)
CORS(app)
try:
    model = YOLO(config.MODEL_PATH)
    print(f"Successfully loaded YOLO model from {config.MODEL_PATH}")
except Exception as e:
    print(f"--- FATAL ERROR: YOLO model failed to load from {config.MODEL_PATH} ---")
    print(f"Make sure '{config.MODEL_PATH}' is in the same directory as this script.")
    print(f"Error details: {e}")
    model = None

parking_status = {"free": 0, "occupied": 0, "direction": "N/A"}
status_lock = threading.Lock()

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_direction_from_angle(angle):
    """Converts a navigation angle (in degrees) to a human-readable string."""
    if angle > 75 and angle < 105:
        return "Forward"
    elif angle > 45 and angle <= 75:
        return "Forward-Right"
    elif angle > 105 and angle <= 135:
        return "Forward-Left"
    elif angle > 15 and angle <= 45:
        return "Sharp Right"
    elif angle > 135 and angle <= 165:
        return "Sharp Left"
    elif angle >= -15 and angle <= 15:
        return "Right"
    elif angle > 165 or angle < -165:
        return "Left"
    elif angle < -15 and angle >= -75:
        return "Reverse-Right"
    elif angle < -105 and angle >= -165:
        return "Reverse-Left"
    elif angle < -75 and angle >= -105:
        return "Reverse"
    else:
        return "N/A" # Should not happen

def process_video_frames(video_path, location_key):
    global parking_status
    entry_point = config.ENTRY_POINTS.get(location_key)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Video not found: {video_path}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {video_path}")
        return

    print(f"Starting video processing for: {video_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"End of video '{video_path}'. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if model:
            results = model(frame, verbose=False)

            free_slots_centers = []
            occupied_count = 0
            direction_message = "N/A"
            nearest_center = None

            annotated_frame = results[0].plot()

            for r in results:
                for box in r.boxes:
                    class_name = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    if class_name == 'free':
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        free_slots_centers.append(center)
                    elif class_name == 'car' or class_name == 'occupied':
                        occupied_count += 1

            free_count = len(free_slots_centers)

            if entry_point and free_slots_centers:
                distances = [
                    (calculate_distance(entry_point, center), center)
                    for center in free_slots_centers
                ]
                nearest_distance, nearest_center = min(distances, key=lambda x: x[0])

                # We use -y_diff because Y-axis is inverted in images (0 is at top)
                x_diff = nearest_center[0] - entry_point[0]
                y_diff = nearest_center[1] - entry_point[1]
                angle = math.degrees(math.atan2(-y_diff, x_diff))

                direction_message = get_direction_from_angle(angle)
                
                cv2.line(annotated_frame, entry_point, nearest_center, (255, 255, 0), 3) # Cyan line
                cv2.circle(annotated_frame, entry_point, 10, (0, 255, 255), -1) # Yellow circle
                cv2.putText(annotated_frame, "ENTRY", (entry_point[0] + 15, entry_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            elif entry_point:
                direction_message = "Lot Full"
                cv2.circle(annotated_frame, entry_point, 10, (0, 0, 255), -1) # Red circle
                cv2.putText(annotated_frame, "LOT FULL", (entry_point[0] + 15, entry_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            with status_lock:
                parking_status["free"] = free_count
                parking_status["occupied"] = occupied_count
                parking_status["direction"] = direction_message

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 50), (0, 0, 0), -1) # Black box
            
            alpha = 0.6 # Transparency
            annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

            cv2.putText(
                annotated_frame, 
                f"NAV: {direction_message.upper()}", 
                (10, 35),  # Position
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0,       # Scale
                (255, 255, 255), # Color
                2          # Thickness
            )

            (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
            if not flag:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        else:
            cv2.putText(frame, "MODEL FAILED TO LOAD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()
    print(f"Released video capture for: {video_path}")

@app.route('/video_feed')
def video_feed():
    location_key = request.args.get('location', 'North_Lot')
    video_path = config.VIDEO_SOURCES.get(location_key)

    if not video_path:
        print(f"Error: Invalid location key received: {location_key}")
        return "Error: Invalid location specified.", 404

    return Response(process_video_frames(video_path, location_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with status_lock:
        current_status = parking_status.copy()

    return jsonify(
        free=current_status["free"], 
        occupied=current_status["occupied"],
        direction=current_status.get("direction", "N/A")
    )

if __name__ == '__main__':
    print("--- Starting SWIFT SLOT AI Server ---")
    print(f"Model: {config.MODEL_PATH}")
    print(f"Available Video Sources: {list(config.VIDEO_SOURCES.keys())}")
    print(f"Configured Entry Points: {config.ENTRY_POINTS}")
    app.run(debug=True, threaded=True, use_reloader=False)
