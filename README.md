# RoboPoint: Vision-Language Model for Spatial Affordance Prediction

## Table of Contents
- [Introduction](#introduction)
- [Quick Start with Docker](#quick-start-with-docker)
- [Running the System](#running-the-system)
- [System Architecture](#system-architecture)

## Introduction

RoboPoint is a Vision-Language Model (VLM) that predicts image keypoint affordances given language instructions. The system uses a distributed architecture with three main components: Controller, Model Worker, and PyQt Server (user interface).

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

### Build the Docker Image

```bash
cd docker
./build.sh
```

This will build the Docker image with all necessary dependencies including CUDA support, PyQt5, and RoboPoint requirements.

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
- Using `--load-4bit` significantly reduces GPU memory requirements (minimum 12GB VRAM)
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
   - Supports multi-GPU distributed deployment
   - Supports 4-bit/8-bit quantization to reduce memory usage

3. **PyQt Server**
   - Provides graphical user interface
   - Supports image upload and camera input
   - Visualizes prediction results
