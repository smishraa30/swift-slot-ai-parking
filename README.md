# ⚡ Swift Slot: AI Smart Parking System

An AI-powered smart parking system using YOLOv8, OpenCV, and Flask to provide real-time slot detection and navigation.

---

## 🚀 Core Features
* **Real-time Object Detection**: Uses a custom-trained YOLOv8 model to detect free (`free`) and occupied (`car`) spots.
* **Live Video Streaming**: The Flask backend (`api_server.py`) streams the processed video feed directly to the web dashboard.
* **Dynamic Dashboard**: The frontend (`features.html`) polls a JSON API (`/api/status`) to display real-time counts.
* **Navigation Guidance**: Calculates the vector from an entry point to the nearest free slot and provides simple directions.
* **Futuristic UI**: A modern frontend built with HTML, Tailwind CSS, and vanilla JavaScript.

---

## 🛠️ Tech Stack
* **Backend**: Python, Flask, Flask-CORS
* **AI/CV**: Ultralytics YOLOv8, OpenCV
* **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript

---

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/swift-slot-ai-parking.git](https://github.com/your-username/swift-slot-ai-parking.git)
    cd swift-slot-ai-parking
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration (Important!)

Before running the server, you **must** configure the entry points for your video feeds.

1.  Open `api_server.py`.
2.  Find the `ENTRY_POINTS` dictionary.
3.  Update the `(x, y)` coordinates for each video source to match the location of the "entry" gate in your video.
    ```python
    # Example:
    ENTRY_POINTS = {
        "North_Lot": (320, 480),  # <-- CHANGE THIS
        "South_Garage": (100, 450), # <-- CHANGE THIS
        "East_Field": (600, 450),  # <-- CHANGE THIS
    }
    ```

---

## ▶️ How to Run

1.  **Start the Flask server:**
    ```bash
    python api_server.py
    ```

2.  **View the application:**
    * This project does not currently serve the frontend. The easiest way is to **open the `features.html` file directly in your browser.**
    * *(See "Professional Improvements" below for how to serve this file from Flask).*
