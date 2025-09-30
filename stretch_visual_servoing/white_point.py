import os
import argparse
import time
from copy import deepcopy

import numpy as np
import cv2
import zmq

import yolo_networking as yn
import loop_timer as lt
import d405_helpers_without_pyrealsense as dh
import aruco_detector as ad
import aruco_to_fingertips as af
import yaml
from yaml.loader import SafeLoader


def robust_depth_at_pixel(depth_image, px, py, depth_scale, k=3):
    h, w = depth_image.shape[:2]
    x0 = max(0, px - k); x1 = min(w, px + k + 1)
    y0 = max(0, py - k); y1 = min(h, py + k + 1)
    patch = depth_image[y0:y1, x0:x1]
    if patch is None or patch.size == 0:
        return None
    vals = patch.reshape(-1)
    # Convert to float meters
    if np.issubdtype(vals.dtype, np.floating):
        vals = vals[np.isfinite(vals)]
        vals_m = vals
    else:
        vals = vals.astype(np.float32)
        vals_m = vals * float(depth_scale)
    vals_m = vals_m[vals_m > 0]
    if vals_m.size == 0:
        return None
    return float(np.median(vals_m))


class WhitePointTracker:
    """Simple template-matching tracker to keep the selected pixel on the same object.

    - Initializes with a template patch around the initial pixel.
    - On each new frame, searches within a local window to find best match.
    - Returns updated (px, py) in current frame coordinates.
    """
    def __init__(self, init_px: int, init_py: int, template_size: int = 41, search_radius: int = 40):
        self.px = int(init_px)
        self.py = int(init_py)
        self.template_size = int(max(11, template_size | 1))  # ensure odd, >=11
        self.search_radius = int(max(8, search_radius))
        self.template = None
        self.initialized = False

    def _extract_patch(self, gray, cx, cy, size):
        h, w = gray.shape[:2]
        half = size // 2
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        y0 = max(0, cy - half); y1 = min(h, cy + half + 1)
        patch = gray[y0:y1, x0:x1]
        # Pad if near borders
        if patch.shape[0] != size or patch.shape[1] != size:
            pad_t = size - patch.shape[0]
            pad_l = size - patch.shape[1]
            patch = cv2.copyMakeBorder(patch, 0, pad_t, 0, pad_l, cv2.BORDER_REPLICATE)
        return patch

    def initialize(self, color_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        self.template = self._extract_patch(gray, self.px, self.py, self.template_size)
        self.initialized = True

    def update(self, color_image):
        if not self.initialized:
            self.initialize(color_image)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        h, w = gray.shape[:2]
        sr = self.search_radius
        half_t = self.template_size // 2
        # Define search window around last position
        x0 = max(0, self.px - sr - half_t); x1 = min(w, self.px + sr + half_t + 1)
        y0 = max(0, self.py - sr - half_t); y1 = min(h, self.py + sr + half_t + 1)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < self.template_size or search.shape[1] < self.template_size:
            # Too close to borders; fall back to previous position
            return self.px, self.py
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Top-left of best match in search image
        top_left = max_loc
        # Compute center of match in full image coordinates
        cx = x0 + top_left[0] + half_t
        cy = y0 + top_left[1] + half_t
        # Update internal state
        self.px, self.py = int(cx), int(cy)
        return self.px, self.py


def main(use_remote_computer: bool, x: int, y: int, push_radius_m: float,
         template_size: int = 41, search_radius: int = 40, extra_push_m: float = 0.02):
    # Publisher for task-relevant features (same port and schema as YOLO module)
    yolo_context = zmq.Context()
    yolo_socket = yolo_context.socket(zmq.PUB)
    if use_remote_computer:
        yolo_address = 'tcp://*:' + str(yn.yolo_port)
    else:
        yolo_address = 'tcp://' + '127.0.0.1' + ':' + str(yn.yolo_port)
    yolo_socket.setsockopt(zmq.SNDHWM, 1)
    yolo_socket.setsockopt(zmq.RCVHWM, 1)
    yolo_socket.bind(yolo_address)

    # Subscriber to D405 image stream
    d405_context = zmq.Context()
    d405_socket = d405_context.socket(zmq.SUB)
    d405_socket.setsockopt(zmq.SUBSCRIBE, b'')
    d405_socket.setsockopt(zmq.SNDHWM, 1)
    d405_socket.setsockopt(zmq.RCVHWM, 1)
    d405_socket.setsockopt(zmq.CONFLATE, 1)
    if use_remote_computer:
        d405_address = 'tcp://' + yn.robot_ip + ':' + str(yn.d405_port)
    else:
        d405_address = 'tcp://' + '127.0.0.1' + ':' + str(yn.d405_port)
    d405_socket.connect(d405_address)

    # ArUco fingertip estimation (for fingertips in send_dict)
    with open('aruco_marker_info.yaml') as f:
        marker_info = yaml.load(f, Loader=SafeLoader)
    aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False,
                                      use_apriltag_refinement=False, brighten_images=False)
    aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height['cup_top'])

    loop_timer = lt.LoopTimer()

    camera_info = None
    depth_scale = None
    tracker = WhitePointTracker(x, y, template_size=template_size, search_radius=search_radius)

    try:
        while True:
            loop_timer.start_of_iteration()

            d405_output = d405_socket.recv_pyobj()
            color_image = d405_output.get('color_image')
            depth_image = d405_output.get('depth_image')
            depth_camera_info = d405_output.get('depth_camera_info')
            depth_scale = d405_output.get('depth_scale')
            if (color_image is None) or (depth_image is None) or (depth_camera_info is None) or (depth_scale is None):
                continue
            camera_info = depth_camera_info

            # Fingertips estimation
            try:
                aruco_detector.update(color_image, camera_info)
                markers = aruco_detector.get_detected_marker_dict()
                fingertips = aruco_to_fingertips.get_fingertips(markers)
            except Exception:
                fingertips = {}

            # Track pixel location to keep dot attached to the object
            h, w = depth_image.shape[:2]
            px, py = tracker.update(color_image)
            px = int(np.clip(px, 0, w - 1))
            py = int(np.clip(py, 0, h - 1))
            z_m = robust_depth_at_pixel(depth_image, px, py, depth_scale, k=3)

            yolo_list = []
            if (z_m is not None) and (z_m > 0):
                center_surf = dh.pixel_to_3d(np.array([px, py], dtype=np.float32), z_m, camera_info)
                grasp_center_xyz = center_surf
                if push_radius_m and push_radius_m > 0:
                    # push along viewing ray to approximate object center
                    ray = center_surf / (np.linalg.norm(center_surf) + 1e-9)
                    grasp_center_xyz = center_surf + (push_radius_m * ray)
                # Always add a small extra forward push if configured (default 2 cm)
                if extra_push_m and extra_push_m > 0:
                    ray = center_surf / (np.linalg.norm(center_surf) + 1e-9)
                    grasp_center_xyz = grasp_center_xyz + (extra_push_m * ray)

                det = {
                    'grasp_center_xyz': grasp_center_xyz,
                    'width_m': float(push_radius_m * 2.0) if push_radius_m and push_radius_m > 0 else 0.0,
                }
                yolo_list.append(det)

            send_dict = {
                'fingertips': fingertips,
                'yolo': yolo_list
            }

            yolo_socket.send_pyobj(send_dict)

            cv2.waitKey(1)
            loop_timer.end_of_iteration()
            loop_timer.pretty_print(minimum=True)

    finally:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='White Point to 3D Grasp Publisher',
        description='Converts a selected white-point pixel + depth to a 3D grasp_center_xyz and publishes on YOLO ZMQ channel.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Run on remote computer (bind publisher, subscribe to robot D405).')
    parser.add_argument('-x', type=int, required=True, help='Pixel x (columns) in the D405 color/depth image space.')
    parser.add_argument('-y', type=int, required=True, help='Pixel y (rows) in the D405 color/depth image space.')
    parser.add_argument('--radius-m', type=float, default=0.0, help='Optional push radius (meters) to move from surface toward object center.')
    parser.add_argument('--template-size', type=int, default=41, help='Template patch size (odd).')
    parser.add_argument('--search-radius', type=int, default=40, help='Search radius in pixels around last position.')
    parser.add_argument('--extra-push-m', type=float, default=0.02, help='Small forward offset along viewing ray (meters). Default 0.02.')
    args = parser.parse_args()

    main(use_remote_computer=args.remote,
         x=args.x,
         y=args.y,
         push_radius_m=args.radius_m,
         template_size=args.template_size,
         search_radius=args.search_radius,
         extra_push_m=args.extra_push_m)
