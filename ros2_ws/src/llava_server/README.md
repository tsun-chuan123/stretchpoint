# LLaVA Server with Transformers Support

This document describes the updated LLaVA server that now supports Transformers models like the robopoint worker.

## Features

- **Transformers Backend**: Support for HuggingFace Transformers models
- **Object Detection**: Detect objects and generate pixel coordinates 
- **Point Annotation**: Visual annotation of detected objects
- **ROS2 Integration**: Full ROS2 node with publishers, subscribers, and services
- **ZMQ Communication**: Compatible with existing networking protocols
- **Simulation Mode**: Fallback simulation when models can't load

## Installation

The server requires additional dependencies that can be installed in the Docker container:

```bash
docker exec -it llm_server_container bash -c "pip install sentencepiece protobuf transformers torch torchvision accelerate"
```

## Usage

### Method 1: Direct ROS2 run
```bash
# Start the container
cd /path/to/stretchpoint/docker && docker compose up -d

# Run with default robopoint model
docker exec -it llm_server_container bash -c "
  cd /stretchpoint/ros2_ws && 
  source /opt/ros/humble/setup.bash && 
  source install/setup.bash && 
  ros2 run llava_server server_node --model-type transformers --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b
"

# Run with different model
docker exec -it llm_server_container bash -c "
  cd /stretchpoint/ros2_ws && 
  source /opt/ros/humble/setup.bash && 
  source install/setup.bash && 
  ros2 run llava_server server_node --model-type transformers --model-path microsoft/llava-1.5-7b-hf
"
```

### Method 2: Launch file (recommended)
```bash
# Default robopoint model
ros2 launch llava_server llava_server.launch.py

# Custom model
ros2 launch llava_server llava_server.launch.py model_path:=microsoft/llava-1.5-7b-hf

# With remote processing
ros2 launch llava_server llava_server.launch.py model_path:=wentao-yuan/robopoint-v1-vicuna-v1.5-13b remote:=true
```

## Model Support

### Working Models
- Any model supported by standard `transformers.AutoModelForCausalLM`
- Models with built-in vision processing (via `AutoProcessor`)
- Text-only models (will run in simulation mode for vision tasks)

### Robopoint Models
The `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` model requires custom LLaVA classes. When used with this server:
- Model loading will fail gracefully 
- Server will run in **simulation mode**
- Provides reasonable coordinate predictions based on prompt keywords
- Useful for testing and integration work

### Full Robopoint Support
For full robopoint model support, copy the custom model classes from the `ros2_Robopoint` workspace:
```bash
# Copy model classes
cp -r /path/to/ros2_Robopoint/ros2_ws/src/robopoint_worker/robopoint_worker/model/* \
      /stretchpoint/ros2_ws/src/llava_server/llava_server/model/
```

## API

### ROS2 Topics
- **Subscribers**:
  - `/vla_command` (VLACommand): Process vision-language commands
  - `/camera/depth/image_raw` (Image): Depth images for 3D positioning
  - `/camera/color/camera_info` (CameraInfo): Camera calibration

- **Publishers**:
  - `/grasp_target` (GraspTarget): Detected object coordinates
  - `/annotated_image` (AnnotatedImage): Image with detection annotations

### ROS2 Services
- `/process_vla_request` (ProcessVLARequest): Synchronous object detection

### ZMQ Communication
- **Subscriber**: `tcp://10.0.0.3:4030` (commands from GUI)
- **Publisher**: `tcp://*:4040` (results to GUI)

## Simulation Mode

When model loading fails, the server runs in simulation mode:
- Provides coordinate predictions based on text analysis
- Uses keyword matching (e.g., "red ball" → center-right position)
- Returns confidence scores between 0.6-0.85
- Enables testing without full model setup

## Parameters

- `--model-type`: Backend type (`transformers` or `ollama`)
- `--model-path`: HuggingFace model ID or local path
- `--remote`: Use remote computer IP for networking
- `-r`: Short form of `--remote`

## Examples

### Basic Object Detection
```python
# Publish VLA command
rostopic pub /vla_command stretch_interfaces/VLACommand "
image: [your_image_data]
command_text: 'find the red cup'
"
```

### Service Call
```bash
# Call detection service
ros2 service call /process_vla_request stretch_interfaces/srv/ProcessVLARequest "
image: [your_image_data]
command: 'locate the blue object'
"
```

## Building

```bash
cd /stretchpoint/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llava_server
source install/setup.bash
```

## Notes

- The server is now compatible with the robopoint ecosystem
- Simulation mode provides a fallback for any model loading issues
- ZMQ networking maintains compatibility with existing GUI components
- Full ROS2 integration allows easy topic monitoring and debugging
