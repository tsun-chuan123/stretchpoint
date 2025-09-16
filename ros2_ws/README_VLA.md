# VLA System with Visual Servoing Architecture

This project implements a Visual Language Action (VLA) system for the Stretch robot using an architecture similar to the visual servoing pattern.

## Architecture Overview

The system follows the visual servoing communication pattern with the following components:

1. **Robot Side (stretchpoint docker)**:
   - `gui_node.py`: PyQt5 GUI for sending natural language commands
   - `send_camera_images.py`: Sends camera data from robot to server
   - Publishes commands on port 4030, receives results on port 4040

2. **Server Side (separate computer)**:
   - `server_node.py`: LLaVA server that processes images and commands
   - Subscribes to camera data (port 4405) and commands (port 4030)
   - Publishes results on port 4040

## Network Configuration

The system uses the same networking pattern as visual servoing:

- **Camera Data**: Port 4405 (robot → server)
- **Commands**: Port 4030 (robot → server)  
- **Results**: Port 4040 (server → robot)

Default IPs (configurable via environment variables):
- Robot: `10.0.0.3`
- Server: `10.0.0.1`

## Usage

### 1. Start the Robot Side (in docker container)

```bash
# Build the workspace
cd /home/hrc/stretchpoint/ros2_ws
colcon build --packages-select stretch_gui llava_server

# Source the workspace
source install/setup.bash

# Start camera image sender and GUI
ros2 launch stretch_gui vla_robot.launch.py use_remote:=true
```

Or run components separately:
```bash
# Send camera images
ros2 run stretch_gui send_camera_images --remote

# Start GUI
ros2 run stretch_gui gui_node --remote
```

### 2. Start the Server Side (on separate computer)

```bash
# On your server computer, build and run:
cd /path/to/your/workspace
colcon build --packages-select llava_server stretch_gui

source install/setup.bash

# Start VLA server
ros2 run llava_server server_node --remote
```

### 3. Test the System

Use the command line test tool:
```bash
# Interactive testing
ros2 run stretch_gui vla_command_test --remote --interactive

# Single command test
ros2 run stretch_gui vla_command_test --remote --command "Pick up the red ball"
```

## Environment Variables

You can override network configuration:

```bash
export VLA_ROBOT_IP=10.0.0.3        # Robot IP
export VLA_SERVER_HOST=10.0.0.1     # Server IP  
export D405_PORT=4405               # Camera data port
export NL_COMMAND_PORT=4030         # Command port
export NL_TARGET_PORT=4040          # Results port
```

## Key Changes from Original

1. **Communication Pattern**: Changed from REQ/REP to PUB/SUB following visual servoing pattern
2. **Camera Data Flow**: Added camera image streaming from robot to server
3. **Command Flow**: Commands published from robot, processed on server
4. **Result Flow**: Processing results streamed back to robot
5. **Loop Timer**: Added performance monitoring similar to visual servoing

## Files Modified

- `server_node.py`: Restructured to follow visual servoing receiver pattern
- `gui_node.py`: Modified to use PUB/SUB communication
- `vla_networking.py`: Updated to match visual servoing port scheme
- Added `send_camera_images.py`: Camera streaming (like `send_d405_images.py`)
- Added `vla_command_test.py`: Command line testing tool
- Added `loop_timer.py`: Performance monitoring

## Running in Your Setup

Since you mentioned running in stretchpoint docker but with server on another computer:

1. **In Docker Container**:
   ```bash
   ./docker/run.sh
   # Inside container:
   cd /home/hrc/stretchpoint/ros2_ws
   colcon build --packages-select stretch_gui
   source install/setup.bash
   ros2 run stretch_gui gui_node --remote
   ros2 run stretch_gui send_camera_images --remote
   ```

2. **On Server Computer**:
   ```bash
   # Copy the llava_server package to your server
   # Build and run:
   ros2 run llava_server server_node --remote
   ```

The `--remote` flag configures the system for cross-host communication using the IPs defined in `vla_networking.py`.
