#!/usr/bin/env python3
"""Networking config for Stretch GUI <-> LLaVA server pipeline.

This module centralizes IP/port configuration based on the visual servoing pattern
and supports environment variable overrides so you can run across hosts 
(e.g., over WireGuard VPN) without editing code elsewhere.

Environment variables (optional):
  - VLA_ROBOT_IP: WireGuard IP of the robot (GUI host) [default: 10.0.0.3]
  - VLA_SERVER_HOST: WireGuard IP/hostname of the LLaVA server [default: 10.0.0.1]
  - VLA_REMOTE_IP: Legacy alias for server host (used if VLA_SERVER_HOST unset)
  - VLA_USE_REMOTE: '1/true' if running across hosts; otherwise local loopback [default: 1]
  - D405_PORT: Depth camera port [default: 4405]
  - YOLO_PORT: YOLO service port [default: 4010]
  - DETECTIONS_PORT: PUB/SUB detections stream [default: 4020]
  - NL_COMMAND_PORT: PUB bind port for NL commands (client publishes) [default: 4030]
  - NL_TARGET_PORT: PUB bind port for NL results (server publishes) [default: 4040]
  - VLA_SERVER_PORT: ZMQ REQ/REP port for LLaVA server [default: 5555]

Usage pattern (similar to visual servoing):
  - GUI (client) publishes NL commands at NL_COMMAND_PORT on the robot host.
  - LLaVA server subscribes to robot_ip:NL_COMMAND_PORT and publishes results
    at NL_TARGET_PORT on the server host.
  - GUI subscribes to server_host:NL_TARGET_PORT.
  - Camera data flows from robot's D405_PORT to server.
"""

import os

# Hosts - match visual servoing pattern
robot_ip = os.environ.get('VLA_ROBOT_IP', '10.0.0.3')
remote_computer_ip = os.environ.get('VLA_SERVER_HOST', os.environ.get('VLA_REMOTE_IP', '10.0.0.1'))

# Ports - using same scheme as visual servoing
# hello d405 => 4ello d405 => 4405
d405_port = int(os.environ.get('D405_PORT', 4405))
# hello YOLO => 4ello Y010 => 4010
yolo_port = int(os.environ.get('YOLO_PORT', 4010))
# Additional PUB/SUB ports for extended pipeline
# Full detections stream (for NL server)
detections_port = int(os.environ.get('DETECTIONS_PORT', 4020))
# Natural-language command input (client publishes here)
nl_command_port = int(os.environ.get('NL_COMMAND_PORT', 4030))
# Selected target output (server publishes here)
nl_target_port = int(os.environ.get('NL_TARGET_PORT', 4040))

# ZMQ REQ/REP port for LLaVA server (backward compatibility)
llava_reqrep_port = int(os.environ.get('VLA_SERVER_PORT', 5555))

# Remote toggle helper
def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() in ('1', 'true', 'yes', 'on')

VLA_USE_REMOTE = _env_bool('VLA_USE_REMOTE', True)

# Convenience helpers (not mandatory; some modules build endpoints inline)
def reqrep_bind_endpoint() -> str:
    """Endpoint for server REQ/REP bind (on the server host)."""
    return f"tcp://*:{llava_reqrep_port}"

def reqrep_connect_endpoint() -> str:
    """Endpoint for GUI REQ/REP connect to server host."""
    host = os.environ.get('VLA_SERVER_HOST', remote_computer_ip)
    return f"tcp://{host}:{llava_reqrep_port}"

def pub_bind_commands_endpoint() -> str:
    """Endpoint for GUI PUB bind for NL commands (on the robot host)."""
    return f"tcp://*:{nl_command_port}"

def sub_connect_commands_endpoint() -> str:
    """Endpoint for server SUB connect to robot host commands PUB."""
    host = os.environ.get('VLA_ROBOT_IP', robot_ip)
    return f"tcp://{host}:{nl_command_port}"

def pub_bind_results_endpoint() -> str:
    """Endpoint for server PUB bind for NL results (on the server host)."""
    return f"tcp://*:{nl_target_port}"

def sub_connect_results_endpoint() -> str:
    """Endpoint for GUI SUB connect to server results PUB."""
    host = os.environ.get('VLA_SERVER_HOST', remote_computer_ip)
    return f"tcp://{host}:{nl_target_port}"

# Local/remote variants (for scripts that support both modes like the reference)
def sub_connect_commands_endpoint_auto(remote: bool | None = None) -> str:
    """If remote, connect to robot_ip; else connect to localhost."""
    if remote is None:
        remote = VLA_USE_REMOTE
    host = robot_ip if remote else '127.0.0.1'
    return f"tcp://{host}:{nl_command_port}"

def sub_connect_results_endpoint_auto(remote: bool | None = None) -> str:
    """If remote, connect to server host; else connect to localhost."""
    if remote is None:
        remote = VLA_USE_REMOTE
    host = remote_computer_ip if remote else '127.0.0.1'
    return f"tcp://{host}:{nl_target_port}"

def reqrep_connect_endpoint_auto(remote: bool | None = None) -> str:
    if remote is None:
        remote = VLA_USE_REMOTE
    host = remote_computer_ip if remote else '127.0.0.1'
    return f"tcp://{host}:{llava_reqrep_port}"

def local_bind(port: int) -> str:
    """Wildcard bind on current host for given port."""
    return f"tcp://*:{port}"

def local_connect(port: int) -> str:
    """Connect to localhost for given port (loopback)."""
    return f"tcp://127.0.0.1:{port}"
