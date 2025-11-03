# ⚡ Swift Slot: AI Smart Parking System

An AI-powered smart parking system using YOLOv8, OpenCV, and Flask to provide real-time slot detection and navigation.

🚀 Core Features
* Real-time Object Detection: Uses a custom-trained YOLOv8 model  to detect free parking spots ('free') and occupied spots ('car').

Live Video Streaming: The Flask backend (api_server.py) streams the processed video feed directly to the web dashboard.

Dynamic Dashboard: The frontend (features.html) polls a JSON API (/api/status) to display real-time counts of free, occupied, and total slots.

Navigation Guidance: The system calculates the vector from a predefined entry point to the nearest free slot and provides simple navigation directions (e.g., "Forward-Right", "Lot Full").

Futuristic UI: A modern, animated frontend built with HTML, Tailwind CSS, and vanilla JavaScript.

🛠️ Tech Stack
Backend: Python, Flask, Flask-CORS

AI/CV: Ultralytics YOLOv8, OpenCV

Frontend: HTML5, Tailwind CSS, Vanilla JavaScript (Async/Await, Fetch API)

🔧 How It Works
The Flask Server (api_server.py) starts and loads the custom YOLOv8 model (best.pt).

It begins processing a video source (e.g., carPark1.mp4) frame by frame.

On each frame, the YOLO model identifies all 'car' and 'free' spots.

The server updates a global status with the counts and calculates the direction to the closest free spot.

The Frontend (features.html) displays the processed video from the /video_feed endpoint.

Simultaneously, a JavaScript function polls the /api/status endpoint every 3 seconds to fetch the latest JSON data and update the dashboard counts and navigation text.
