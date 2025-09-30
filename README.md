# RoboPoint: Vision-Language Model for Spatial Affordance Prediction

*Precise action guidance with image-based keypoint affordance conditioned on language instructions.*

[[Project Page](https://robo-point.github.io)] [[Paper](https://arxiv.org/pdf/2406.10721)]

![Overview](figures/overview.gif)

## Table of Contents
- [Introduction](#introduction)
- [Quick Start with Docker](#quick-start-with-docker)
- [Running the System](#running-the-system)
- [System Architecture](#system-architecture)
- [Advanced Configuration](#advanced-configuration)
- [Model Zoo](#model-zoo)

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

## Advanced Configuration

### Network Configuration

**For localhost deployment**, use `localhost` or `127.0.0.1` for all IP addresses:

```bash
# Controller
python3 -m robopoint.serve.controller --host 0.0.0.0 --port 11000

# Model Worker
python3 -m robopoint.serve.model_worker \
    --host 0.0.0.0 \
    --port 22000 \
    --worker-address http://localhost:22000 \
    --controller-address http://localhost:11000 \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --load-4bit

# PyQt Server
python3 -m robopoint.serve.pyqt_server --controller-url http://localhost:11000
```

**For LAN or multi-machine deployment**:
- Replace `10.0.0.1` with your actual server IP address
- Ensure firewall allows traffic on ports 11000 and 22000

### GPU Memory Requirements

| Configuration | VRAM Required | Description |
|--------------|---------------|-------------|
| Full Precision (FP16) | ~26GB | Best accuracy, requires high-end GPU |
| 8-bit Quantization | ~14GB | Balance between accuracy and performance |
| 4-bit Quantization | ~12GB | Minimum requirement, suitable for most GPUs |

### Multi-GPU Deployment

If you have multiple GPUs, the system will automatically distribute the load:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m robopoint.serve.model_worker \
    --host 0.0.0.0 \
    --port 22000 \
    --worker-address http://10.0.0.1:22000 \
    --controller-address http://10.0.0.1:11000 \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b
```

### Docker Management

**Stop the container:**
```bash
cd docker
./stop.sh
```

**Rebuild the image:**
```bash
cd docker
./build.sh
```

**Access the container as root:**
```bash
cd docker
./run.sh --root
```

### Troubleshooting

**Q: Model download is slow?**  
A: You can pre-download the model from HuggingFace and use a local path:
```bash
--model-path /path/to/local/model
```

**Q: Out of GPU memory?**  
A: Use the `--load-4bit` parameter to enable 4-bit quantization, which minimizes VRAM usage.

**Q: How to verify services are running?**  
A: Each service will display `Uvicorn running on ...` when ready. The PyQt Server will open a GUI window.

**Q: Can I run multiple model workers?**  
A: Yes! Just use different ports and worker-address for each worker.

**Q: GUI doesn't display?**  
A: Ensure X11 forwarding is configured:
```bash
xhost +local:docker
```

## Model Zoo

| Version | Base LLM | Size | HuggingFace Link |
|---------|----------|------|------------------|
| robopoint-v1-vicuna-v1.5-13b | Vicuna-v1.5 | 13B | [Download](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b) |
| robopoint-v1-llama-2-13b | Llama-2 | 13B | [Download](https://huggingface.co/wentao-yuan/robopoint-v1-llama-2-13b) |
| robopoint-v1-vicuna-v1.5-7b-lora | Vicuna-v1.5 | 7B | [Download](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora) |

## Citation

If you find RoboPoint useful for your research and applications, please cite our paper:

```bibtex
@article{yuan2024robopoint,
  title={RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics},
  author={Yuan, Wentao and Duan, Jiafei and Blukis, Valts and Pumacay, Wilbert and Krishna, Ranjay and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  journal={arXiv preprint arXiv:2406.10721},
  year={2024}
}
```

## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA): The codebase we built upon, including the visual instruction tuning pipeline
