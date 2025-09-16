#!/bin/bash

# VLA Server Startup Script
# Run this on your separate server computer

echo "Starting VLA Server..."

# Check if required environment variables are set
if [ -z "$VLA_ROBOT_IP" ]; then
    export VLA_ROBOT_IP="10.0.0.3"
    echo "Using default robot IP: $VLA_ROBOT_IP"
fi

if [ -z "$VLA_SERVER_HOST" ]; then
    export VLA_SERVER_HOST="10.0.0.1"
    echo "Using default server IP: $VLA_SERVER_HOST"
fi

# Function to check if workspace exists
check_workspace() {
    if [ ! -f "install/setup.bash" ]; then
        echo "Error: ROS2 workspace not found or not built"
        echo "Please ensure you've copied the llava_server package and run:"
        echo "  colcon build --packages-select llava_server"
        exit 1
    fi
}

# Function to start server
start_server() {
    echo "Starting VLA server with configuration:"
    echo "  Robot IP: $VLA_ROBOT_IP"
    echo "  Server IP: $VLA_SERVER_HOST"
    echo "  Camera port: ${D405_PORT:-4405}"
    echo "  Command port: ${NL_COMMAND_PORT:-4030}"
    echo "  Result port: ${NL_TARGET_PORT:-4040}"
    
    # Source the workspace
    source install/setup.bash
    
    # Start the server
    ros2 run llava_server server_node --remote
}

# Function to show help
show_help() {
    echo "VLA Server Startup Script"
    echo ""
    echo "Usage: $0 [start|help]"
    echo ""
    echo "Commands:"
    echo "  start (default): Start the VLA server"
    echo "  help: Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VLA_ROBOT_IP: IP address of the robot (default: 10.0.0.3)"
    echo "  VLA_SERVER_HOST: IP address of this server (default: 10.0.0.1)"
    echo "  D405_PORT: Camera data port (default: 4405)"
    echo "  NL_COMMAND_PORT: Command input port (default: 4030)"
    echo "  NL_TARGET_PORT: Result output port (default: 4040)"
    echo ""
    echo "Example:"
    echo "  VLA_ROBOT_IP=192.168.1.100 $0 start"
}

# Main script
case "${1:-start}" in
    "start")
        check_workspace
        start_server
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
