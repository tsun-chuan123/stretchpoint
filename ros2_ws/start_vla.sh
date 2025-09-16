#!/bin/bash

# VLA System Startup Script
# Based on visual servoing pattern

echo "Starting VLA System Components..."

# Check if we're in the docker environment
if [ ! -d "/home/hrc/stretchpoint/ros2_ws" ]; then
    echo "Error: This script should be run in the stretchpoint docker environment"
    exit 1
fi

cd /home/hrc/stretchpoint/ros2_ws

# Build the packages
echo "Building packages..."
colcon build --packages-select stretch_gui llava_server

# Source the workspace
source install/setup.bash

# Function to start robot side components
start_robot() {
    echo "Starting robot side components..."
    
    # Start camera image sender in background
    echo "Starting camera image sender..."
    ros2 run stretch_gui send_camera_images --remote &
    CAMERA_PID=$!
    
    # Start GUI
    echo "Starting GUI..."
    ros2 run stretch_gui gui_node --remote &
    GUI_PID=$!
    
    # Function to cleanup on exit
    cleanup() {
        echo "Stopping components..."
        kill $CAMERA_PID $GUI_PID 2>/dev/null
        exit 0
    }
    
    # Trap signals for cleanup
    trap cleanup SIGINT SIGTERM
    
    echo "Robot components started. PIDs: Camera=$CAMERA_PID, GUI=$GUI_PID"
    echo "Press Ctrl+C to stop all components"
    wait
}

# Function to test the system
test_system() {
    echo "Testing VLA system..."
    ros2 run stretch_gui vla_command_test --remote --interactive
}

# Check command line arguments
case "${1:-robot}" in
    "robot")
        start_robot
        ;;
    "test")
        test_system
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [robot|test|help]"
        echo "  robot (default): Start robot side components"
        echo "  test: Run interactive command test"
        echo "  help: Show this help message"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
