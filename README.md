# RoboPoint: Vision-Language Model for Spatial Affordance Prediction

## Table of Contents
- [Introduction](#introduction)
- [Quick Start with Docker](#quick-start-with-docker)
- [Running the System](#running-the-system)
- [Running Visual Servoing GUI](#running-visual-servoing-gui)
- [System Architecture](#system-architecture)

## Introduction

RoboPoint is a Vision-Language Model (VLM) that predicts image keypoint affordances given language instructions. The system uses a distributed architecture with three main components: Controller, Model Worker, and PyQt Server (user interface).

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- X11 forwarding configured (for GUI display)

### Build the Docker Image

```bash
cd docker
./build.sh
```

This will build the Docker image with all necessary dependencies including PyQt5 and RoboPoint requirements.

## Running the System

The RoboPoint system requires three services to be running: Controller, Model Worker, and PyQt Server (UI). You need to start them in separate terminals.

### Step 1: Enter the Docker Container

```bash
cd docker
./run.sh
```

This will start the container and give you a shell inside it. For subsequent terminals, use the same command to attach to the running container.

### Step 2: Start the Controller

In the first terminal inside the container:

```bash
python3 -m robopoint.serve.controller --host 0.0.0.0 --port 11000
```

Wait until you see `Uvicorn running on ...` message. The controller is now ready.

### Step 3: Start the Model Worker

Open a second terminal and enter the container:

```bash
cd docker
./run.sh
```

Then start the model worker:

```bash
python3 -m robopoint.serve.model_worker \
    --host 0.0.0.0 \
    --port 22000 \
    --worker-address http://10.0.0.1:22000 \
    --controller-address http://10.0.0.1:11000 \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --load-4bit
```

**Important Notes:**
- Replace `10.0.0.1` with your actual server IP address
- First run will automatically download the model from HuggingFace (~13GB)
- Using `--load-4bit` reduces memory requirements
- Wait for model loading to complete and see `Uvicorn running on ...` message

### Step 4: Start the PyQt Server (UI)

Open a third terminal and enter the container:

```bash
cd docker
./run.sh
```

Then start the PyQt server:

```bash
python3 -m robopoint.serve.pyqt_server --controller-url http://10.0.0.1:11000
```

**Important Notes:**
- Replace `10.0.0.1` with the controller's server IP address
- The GUI will launch and you can upload images and input instructions for inference
- Make sure X11 forwarding is properly configured for GUI display

## Running Visual Servoing GUI

The RoboPoint Visual Servoing GUI provides an integrated interface for robot control, combining LLM-based affordance prediction with real-time camera feeds and robot manipulation.

### Local Operation (Direct Camera Access)

Run the visual servoing GUI directly on the robot:

```bash
python3 -m stretch_visual_servoing.robopoint_visual_servoing
```

This mode:
- Connects directly to D405/D435i cameras
- Runs YOLO object detection locally
- Controls robot arm and base navigation
- Provides LLM-guided grasping and navigation

### Remote Operation

Run the GUI on a remote workstation while connecting to the robot's camera stream:

```bash
python3 -m stretch_visual_servoing.robopoint_visual_servoing --remote
```

**On the robot**, you need to start the camera publisher:

```bash
# For D405 camera
python3 -m stretch_visual_servoing.send_d405_images

# For D435i camera
python3 -m stretch_visual_servoing.send_d435i_images
```

### Visual Servoing Features

1. **LLM-Guided Grasping**
   - Upload image or use live camera feed
   - Provide natural language instruction (e.g., "grasp the cup")
   - LLM predicts grasp point on the image
   - Click to manually adjust grasp point
   - Execute grasp with "Start LLM Grasping" button

2. **YOLO-Based Grasping**
   - Automatic object detection using YOLOv8
   - Real-time bounding box visualization
   - Click "Start YOLO Grasping" to execute

3. **LLM-Guided Navigation**
   - Provide navigation instruction (e.g., "move to the table")
   - LLM predicts navigation target
   - Robot autonomously navigates to target
   - Supports head tilt control during navigation

4. **Camera Controls**
   - Switch between D405 (wrist) and D435i (head) cameras
   - Adjust exposure settings (low/medium/high)
   - Real-time depth visualization
   - ArUco marker detection for fingertip tracking

### Configuration Options

The visual servoing GUI supports various command-line options:

```bash
python3 -m stretch_visual_servoing.robopoint_visual_servoing \
    --controller-url http://10.0.0.1:11000 \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --load-4bit \
    --remote
```

- `--controller-url`: URL of the RoboPoint controller service
- `--model-path`: Path to the RoboPoint model
- `--load-4bit`: Enable 4-bit quantization for lower memory usage
- `--remote`: Connect to remote robot camera stream

## System Architecture

```
┌─────────────────┐
│  PyQt Server    │ ← User Interface (default port)
│  (UI)           │
└────────┬────────┘
         │
         ↓ HTTP Requests
┌─────────────────┐
│   Controller    │ ← Coordinator (port 11000)
│  (Coordinator)  │
└────────┬────────┘
         │
         ↓ Task Distribution
┌─────────────────┐
│  Model Worker   │ ← Inference Engine (port 22000)
│  (Inference)    │
└─────────────────┘
```

### Component Overview

1. **Controller**
   - Coordinates multiple model workers
   - Manages request queues
   - Monitors worker health status

2. **Model Worker**
   - Performs actual model inference
   - Supports distributed deployment
   - Supports 4-bit/8-bit quantization to reduce memory usage

3. **PyQt Server**
   - Provides graphical user interface
   - Supports image upload and camera input
   - Visualizes prediction results

4. **Visual Servoing GUI**
   - Integrated robot control interface
   - Real-time camera feed with LLM/YOLO integration
   - Supports grasping and navigation tasks
   - Flexible local/remote operation modes
