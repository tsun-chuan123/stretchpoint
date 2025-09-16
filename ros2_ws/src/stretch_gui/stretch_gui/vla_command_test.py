#!/usr/bin/env python3
"""
Simple command line tool for testing VLA server
Similar to visual servoing demo scripts
"""

import zmq
import json
import argparse
import time
import cv2
import numpy as np
try:
    from stretch_gui import vla_networking as vn
except ImportError:
    # Fallback - create minimal networking config
    import os
    class NetworkingConfig:
        robot_ip = os.environ.get('VLA_ROBOT_IP', '10.0.0.3')
        remote_computer_ip = os.environ.get('VLA_SERVER_HOST', '10.0.0.1')
        d405_port = int(os.environ.get('D405_PORT', 4405))
        nl_command_port = int(os.environ.get('NL_COMMAND_PORT', 4030))
        nl_target_port = int(os.environ.get('NL_TARGET_PORT', 4040))
    vn = NetworkingConfig()


def main():
    parser = argparse.ArgumentParser(
        prog='VLA Command Test',
        description='Send natural language commands to VLA server for testing.'
    )
    parser.add_argument('-r', '--remote', action='store_true',
                       help='Connect to remote VLA server instead of localhost')
    parser.add_argument('-c', '--command', type=str, default="Pick up the red ball",
                       help='Natural language command to send')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - keep asking for commands')
    
    args = parser.parse_args()
    
    # Setup ZMQ connections
    context = zmq.Context()
    
    # Publisher for commands (robot side)
    command_pub = context.socket(zmq.PUB)
    command_pub.setsockopt(zmq.SNDHWM, 1)
    command_pub.setsockopt(zmq.RCVHWM, 1)
    command_address = 'tcp://*:' + str(vn.nl_command_port)
    command_pub.bind(command_address)
    print(f"Command publisher bound to {command_address}")
    
    # Subscriber for results (from server)
    result_sub = context.socket(zmq.SUB)
    result_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    result_sub.setsockopt(zmq.SNDHWM, 1)
    result_sub.setsockopt(zmq.RCVHWM, 1)
    result_sub.setsockopt(zmq.CONFLATE, 1)
    result_sub.RCVTIMEO = 10000  # 10 second timeout
    
    if args.remote:
        result_address = 'tcp://' + vn.remote_computer_ip + ':' + str(vn.nl_target_port)
    else:
        result_address = 'tcp://127.0.0.1:' + str(vn.nl_target_port)
    
    result_sub.connect(result_address)
    print(f"Result subscriber connected to {result_address}")
    
    # Give time for connections to establish
    time.sleep(0.5)
    
    def send_command(command_text):
        """Send a command and wait for response"""
        print(f"\nSending command: '{command_text}'")
        
        # Create command message
        message = {
            "command": command_text,
            "timestamp": int(time.time() * 1000)
        }
        
        # Send command
        command_pub.send_string(json.dumps(message))
        print("Command sent, waiting for response...")
        
        # Wait for response
        try:
            response = result_sub.recv_pyobj()
            print(f"Response received:")
            print(f"  Object: {response.get('object_description', 'N/A')}")
            print(f"  Grasp point: {response.get('grasp_point_2d', 'N/A')}")
            print(f"  Confidence: {response.get('confidence', 'N/A')}")
            print(f"  Reasoning: {response.get('reasoning', 'N/A')}")
            return True
            
        except zmq.Again:
            print("Timeout - no response received")
            return False
        except Exception as e:
            print(f"Error receiving response: {e}")
            return False
    
    try:
        if args.interactive:
            print("Interactive mode - type 'quit' to exit")
            while True:
                try:
                    command = input("\nEnter command: ").strip()
                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    if command:
                        send_command(command)
                except KeyboardInterrupt:
                    break
        else:
            # Single command mode
            send_command(args.command)
            
    finally:
        # Cleanup
        command_pub.close()
        result_sub.close()
        context.term()
        print("\nClosed connections")


if __name__ == '__main__':
    main()
