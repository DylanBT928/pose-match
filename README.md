# Pose Match

Pose Match is a web application that uses pose estimation to match user poses against reference images. It leverages Ultralytics YOLOv8 for pose detection and provides a simple Flask backend with a static frontend.

## Features

- Upload or use webcam images to match against reference poses
- Backend powered by Flask and YOLOv8 pose model
- Simple frontend served via Python HTTP server or Live Server extension

## Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip, wheel, and setuptools

```bash
pip install --upgrade pip wheel setuptools
```

### 3. Install Required Python Packages

```bash
pip install "ultralytics==8.3.34" "opencv-python==4.10.0.84" "numpy>=1.26,<2.1" \
			"flask==3.0.3" "flask-cors==4.0.1"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Running the Application

### Backend

Start the Flask backend server:

```bash
python server.py
```

### Frontend

You can serve the frontend in two ways:

#### Option 1: Python HTTP Server

```bash
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

#### Option 2: Live Server VS Code Extension

Use the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension in VS Code to serve `index.html`.

---

## Project Structure

- `server.py` — Flask backend server
- `index.html` — Frontend HTML file

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
