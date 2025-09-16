#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from stretch_interfaces.msg import VLACommand, GraspTarget, AnnotatedImage
from stretch_interfaces.srv import ProcessVLARequest
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import zmq
import json
import threading
import time
import argparse
import base64
import io
import requests
from PIL import Image as PILImage
import torch
from typing import Optional, Dict, Any, List

try:
    from .loop_timer import LoopTimer
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from loop_timer import LoopTimer

try:
    from stretch_gui import vla_networking as vn
except ImportError:
    # Fallback - create minimal networking config
    class NetworkingConfig:
        robot_ip = '10.0.0.3'
        remote_computer_ip = '10.0.0.1'
        d405_port = 4405
        nl_command_port = 4030
        nl_target_port = 4040
    vn = NetworkingConfig()

# Import LLaVA modules first
try:
    import sys
    sys.path.insert(0, '/stretchpoint/ros2_ws/src/llava_server')
    
    import llava.model.language_model.llava_llama as llava_llama_module
    LlavaLlamaForCausalLM = llava_llama_module.LlavaLlamaForCausalLM
    LlavaConfig = llava_llama_module.LlavaConfig
    
    from llava.model.builder import load_pretrained_model as llava_load_pretrained_model
    from llava.mm_utils import process_images as llava_process_images, tokenizer_image_token as llava_tokenizer_image_token
    from llava.mm_utils import load_image_from_base64 as llava_load_image_from_base64
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
    # 註冊 LLaVA 架構
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register('llava_llama', LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
    
    LLAVA_AVAILABLE = True
    print("LLaVA modules imported and registered successfully")
except ImportError as e:
    print(f"LLaVA import error: {e}")
    LLAVA_AVAILABLE = False

# Import our model utilities as fallback
try:
    from .model.builder import load_pretrained_model
    from .mm_utils import process_images, tokenizer_image_token, load_image_from_base64
    from .mm_utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from model.builder import load_pretrained_model
    from mm_utils import process_images, tokenizer_image_token, load_image_from_base64
    from mm_utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LLaVAModelWorker:
    """
    Model worker for multimodal LLaVA models using Transformers
    Similar to robopoint_worker design but simplified for object detection
    """
    def __init__(self, model_type="transformers", model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", device="cuda"):
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.model_loaded = False
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = 2048
        
        # 初始化模型
        self.load_model()
        
    def load_model(self):
        """載入模型 - 支援多種後端"""
        try:
            if self.model_type == "transformers":
                self.load_transformers_model()
            elif self.model_type == "ollama":
                self.load_ollama_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            self.model_loaded = True
            print(f"Model {self.model_path} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def load_transformers_model(self):
        """載入 Transformers 模型 (優先使用 LLaVA)"""
        try:
            print(f"Loading Transformers model: {self.model_path}")
            
            # 首先嘗試使用 LLaVA 的載入函數
            if LLAVA_AVAILABLE and 'robopoint' in self.model_path.lower():
                try:
                    print("Attempting to load with LLaVA loader...")
                    result = llava_load_pretrained_model(
                        model_path=self.model_path,
                        model_base=None,
                        model_name=self.model_path.split('/')[-1],
                        load_8bit=False,
                        load_4bit=False,
                        device=self.device,
                        use_flash_attn=False
                    )
                    print("LLaVA loader succeeded!")
                except Exception as llava_error:
                    print(f"LLaVA loader failed: {llava_error}")
                    print("Falling back to custom loader...")
                    # 回退到我們的自定義載入器
                    result = load_pretrained_model(
                        model_path=self.model_path,
                        model_base=None,
                        model_name=self.model_path.split('/')[-1],
                        load_8bit=False,
                        load_4bit=False,
                        device=self.device,
                        use_flash_attn=False
                    )
            else:
                # 使用我們的 load_pretrained_model 函數
                result = load_pretrained_model(
                    model_path=self.model_path,
                    model_base=None,
                    model_name=self.model_path.split('/')[-1],
                    load_8bit=False,
                    load_4bit=False,
                    device=self.device,
                    use_flash_attn=False
                )
            
            # 解包結果
            if len(result) == 4:
                self.tokenizer, self.model, self.image_processor, self.context_len = result
                self.processor = None  # 初始化 processor
            elif len(result) == 5:
                self.tokenizer, self.model, self.image_processor, self.context_len, self.processor = result
            else:
                self.tokenizer, self.model, self.image_processor, self.context_len = result[:4]
                self.processor = None
            
            print("Transformers model loaded successfully")
            print(f"Context length: {self.context_len}")
            print(f"Model type: {type(self.model)}")
            print(f"Has vision tower: {hasattr(self.model, 'get_vision_tower')}")
            
        except Exception as e:
            print(f"Failed to load transformers model: {e}")
            print("Will use simulation mode")
            raise
    
    def load_ollama_model(self):
        """載入 Ollama 模型"""
        self.base_url = "http://127.0.0.1:11434"
        self.timeout = 20.0
        
        # 測試 Ollama 連接
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            print("Ollama connection successful")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            raise
    
    def detect_object_with_transformers(self, image, prompt):
        """使用 Transformers 模型進行物件檢測和點生成"""
        try:
            if not self.model_loaded:
                return self.simulate_detection(prompt, image.shape[:2])
            
            # 轉換圖像格式
            img_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 構建提示詞 - 專門用於物件檢測和點生成
            vision_prompt = f"""
USER: <image>
Find the {prompt} in this image and provide the pixel coordinates where it is located.

Please respond with coordinates in the format: (x, y)
Where x and y are the pixel coordinates of the center of the object.

ASSISTANT: Looking at the image, I can see the {prompt}. The coordinates are: """
            
            # 檢查模型類型並使用適當的生成方法
            if hasattr(self.model, 'generate') and hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'model_type') and self.model.config.model_type == 'llava':
                    # LLaVA 模型
                    return self.generate_with_llava_model(img_pil, vision_prompt, image.shape[:2])
                else:
                    # 其他模型 (如 robopoint)
                    return self.generate_with_causal_model(img_pil, vision_prompt, image.shape[:2])
            else:
                return self.simulate_detection(prompt, image.shape[:2])
                
        except Exception as e:
            print(f"Transformers detection error: {e}")
            return self.simulate_detection(prompt, image.shape[:2])
    
    def generate_with_llava_model(self, img_pil, prompt, image_shape):
        """使用 LlavaForConditionalGeneration 生成"""
        try:
            from transformers import LlavaProcessor
            
            # 如果有 processor，使用它
            if hasattr(self, 'processor') and self.processor:
                inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt").to(self.device)
            else:
                # 手動處理
                if self.image_processor:
                    pixel_values = process_images([img_pil], self.image_processor, self.model.config)
                    pixel_values = pixel_values.to(self.device, dtype=torch.float16)
                    
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                    
                    inputs = {
                        "input_ids": input_ids,
                        "pixel_values": pixel_values
                    }
                else:
                    return self.simulate_detection(prompt.split()[-1], image_shape)
            
            # 生成
            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            # 解碼
            generated_text = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"LLaVA generated: {generated_text}")
            
            # 解析座標
            return self.parse_coordinates(generated_text, image_shape, prompt)
            
        except Exception as e:
            print(f"LLaVA generation error: {e}")
            return self.simulate_detection(prompt.split()[-1], image_shape)
    
    def generate_with_causal_model(self, img_pil, prompt, image_shape):
        """使用 AutoModelForCausalLM 生成（如 robopoint 模型）"""
        try:
            # 處理圖像
            if self.image_processor:
                try:
                    images = process_images([img_pil], self.image_processor, self.model.config)
                    if isinstance(images, list):
                        images = [img.to(self.device, dtype=torch.float16) for img in images]
                    else:
                        images = images.to(self.device, dtype=torch.float16)
                except Exception as e:
                    print(f"Image processing error: {e}")
                    images = None
            else:
                images = None
            
            # Token化提示詞
            if self.tokenizer:
                try:
                    # 處理圖像標記
                    replace_token = DEFAULT_IMAGE_TOKEN
                    if hasattr(self.model.config, 'mm_use_im_start_end') and self.model.config.mm_use_im_start_end:
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    
                    if DEFAULT_IMAGE_TOKEN in prompt:
                        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                    
                    # Token化
                    input_ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).to(self.device)
                    
                    # 生成
                    with torch.inference_mode():
                        generate_kwargs = {
                            'input_ids': input_ids,
                            'do_sample': False,
                            'temperature': 0.1,
                            'max_new_tokens': 64,
                            'use_cache': True,
                        }
                        
                        if images is not None:
                            generate_kwargs['images'] = images
                        
                        output_ids = self.model.generate(**generate_kwargs)
                    
                    # 解碼輸出
                    input_token_len = input_ids.shape[1]
                    generated_ids = output_ids[0][input_token_len:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    print(f"Causal model generated: {generated_text}")
                    
                    # 解析座標
                    return self.parse_coordinates(generated_text, image_shape, prompt)
                    
                except Exception as e:
                    print(f"Causal generation error: {e}")
                    return self.simulate_detection(prompt.split()[-1], image_shape)
            else:
                return self.simulate_detection(prompt.split()[-1], image_shape)
                
        except Exception as e:
            print(f"Causal model error: {e}")
            return self.simulate_detection(prompt.split()[-1], image_shape)
    
    def parse_coordinates(self, generated_text, image_shape, prompt):
        """解析生成的座標文字"""
        try:
            height, width = image_shape
            
            # 尋找座標模式: [(x, y)] 或 (x, y) 或 x, y
            import re
            
            # 模式1: [(x, y), (x2, y2), ...]
            coord_pattern1 = r'\[\s*\((\d+),\s*(\d+)\)\s*(?:,\s*\((\d+),\s*(\d+)\))?\s*\]'
            match1 = re.search(coord_pattern1, generated_text)
            
            if match1:
                x, y = int(match1.group(1)), int(match1.group(2))
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                return {
                    'center_px': [x, y],
                    'bbox': [max(0, x-30), max(0, y-30), 60, 60],
                    'confidence': 0.85,
                    'description': f'target from model: {prompt}'
                }
            
            # 模式2: (x, y)
            coord_pattern2 = r'\((\d+),\s*(\d+)\)'
            match2 = re.search(coord_pattern2, generated_text)
            
            if match2:
                x, y = int(match2.group(1)), int(match2.group(2))
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                return {
                    'center_px': [x, y],
                    'bbox': [max(0, x-30), max(0, y-30), 60, 60],
                    'confidence': 0.80,
                    'description': f'target from model: {prompt}'
                }
            
            # 模式3: 數字對 x, y 或 x y
            number_pattern = r'(\d+)[,\s]+(\d+)'
            match3 = re.search(number_pattern, generated_text)
            
            if match3:
                x, y = int(match3.group(1)), int(match3.group(2))
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                return {
                    'center_px': [x, y],
                    'bbox': [max(0, x-30), max(0, y-30), 60, 60],
                    'confidence': 0.75,
                    'description': f'target from model: {prompt}'
                }
            
            # 如果找不到座標，使用中心點
            print(f"Could not parse coordinates from: {generated_text}")
            return {
                'center_px': [width//2, height//2],
                'bbox': [width//2-30, height//2-30, 60, 60],
                'confidence': 0.60,
                'description': f'fallback center: {prompt}'
            }
            
        except Exception as e:
            print(f"Coordinate parsing error: {e}")
            return self.simulate_detection(prompt, image_shape)
    
    def detect_object_with_ollama(self, image, prompt):
        """使用 Ollama 進行物件檢測"""
        try:
            # 編碼圖片為 base64
            img_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG", quality=85)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            # 構建請求
            system_prompt = (
                "You are a vision assistant for robot control. Given an image and a command, "
                "identify the target object and return ONLY a JSON response with: "
                "{'center_px': [x, y], 'bbox': [x, y, w, h], 'confidence': 0.0-1.0, 'description': 'object name'}. "
                "The coordinates should be pixel positions (integers). Origin is top-left. "
                "If uncertain or object not found, return {'center_px': null, 'confidence': 0.0}."
            )
            
            payload = {
                "model": self.model_path,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": b64_img},
                            {"type": "text", "text": f"Find and locate: {prompt}"}
                        ]
                    }
                ],
                "options": {"temperature": 0.0}
            }
            
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            
            if not content:
                return None
                
            # 解析 JSON 回應
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json\n"):
                    content = content[5:]
                    
            result = json.loads(content)
            return result
            
        except Exception as e:
            print(f"Ollama detection error: {e}")
            return self.simulate_detection(prompt, image.shape[:2])
    
    def simulate_detection(self, prompt, image_shape):
        """模擬物件檢測 (用於測試或備用)"""
        height, width = image_shape
        
        # 基於關鍵字的簡單模擬
        if any(word in prompt.lower() for word in ['red', 'ball', 'circle']):
            return {
                'center_px': [width//2 + 50, height//2],
                'bbox': [width//2 + 25, height//2 - 25, 50, 50],
                'confidence': 0.85,
                'description': 'red ball'
            }
        elif any(word in prompt.lower() for word in ['cup', 'mug', 'bottle']):
            return {
                'center_px': [width//3, height//2],
                'bbox': [width//3 - 30, height//2 - 40, 60, 80],
                'confidence': 0.75,
                'description': 'cup'
            }
        else:
            return {
                'center_px': [width//2, height//2],
                'bbox': [width//2 - 40, height//2 - 40, 80, 80],
                'confidence': 0.60,
                'description': 'target object'
            }
    
    def detect_object(self, image, prompt):
        """統一的物件檢測介面"""
        if not self.model_loaded:
            return self.simulate_detection(prompt, image.shape[:2])
            
        if self.model_type == "transformers":
            return self.detect_object_with_transformers(image, prompt)
        elif self.model_type == "ollama":
            return self.detect_object_with_ollama(image, prompt)
        else:
            return self.simulate_detection(prompt, image.shape[:2])


class ObjectAnnotator:
    """物件標注管理器 - 將檢測結果繪製到圖片上"""
    
    def __init__(self):
        pass
        
    def annotate_image(self, image, detection_results):
        """在圖片上標注檢測結果"""
        annotated_image = image.copy()
        grasp_targets = []
        
        if not isinstance(detection_results, list):
            detection_results = [detection_results] if detection_results else []
        
        for detection in detection_results:
            if detection and detection.get('center_px') and detection['center_px'] is not None:
                center_px = detection['center_px']
                confidence = detection.get('confidence', 0.0)
                description = detection.get('description', 'unknown object')
                bbox = detection.get('bbox')
                
                if isinstance(center_px, list) and len(center_px) == 2:
                    x, y = int(center_px[0]), int(center_px[1])
                    
                    # 繪製檢測點 (綠色圓圈)
                    cv2.circle(annotated_image, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(annotated_image, (x, y), 12, (0, 255, 0), 2)
                    
                    # 繪製邊界框 (如果有的話)
                    if bbox and len(bbox) == 4:
                        bx, by, bw, bh = [int(v) for v in bbox]
                        cv2.rectangle(annotated_image, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                    
                    # 添加文字標籤
                    label = f"{description} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # 背景矩形
                    cv2.rectangle(annotated_image, (x - 5, y - 25), 
                                (x + label_size[0] + 5, y - 5), (0, 255, 0), -1)
                    
                    # 文字
                    cv2.putText(annotated_image, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # 創建 grasp target 結構
                    grasp_target = {
                        'pixel_x': float(x),
                        'pixel_y': float(y),
                        'confidence': confidence,
                        'description': description
                    }
                    grasp_targets.append(grasp_target)
        
        return annotated_image, grasp_targets


class LLaVAServerNode(Node):
    """
    參考 robopoint_worker 設計的 LLaVA 服務節點
    提供物件檢測和點標注功能
    """
    def __init__(self, model_type="transformers", model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", use_remote_computer=True):
        super().__init__('llava_server_node')
        
        self.model_type = model_type
        self.model_path = model_path
        self.use_remote_computer = use_remote_computer
        
        # 初始化模型工作器
        self.get_logger().info(f'Initializing LLaVA model worker: {model_type}:{model_path}')
        self.model_worker = LLaVAModelWorker(model_type, model_path)
        
        # 初始化標注器
        self.annotator = ObjectAnnotator()
        
        # ROS2 組件
        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # 初始化 publishers and subscribers
        self.setup_ros_interfaces()
        
        # 設置 ZMQ 通信 (類似原本的設計)
        self.setup_zmq_connections()
        
        # 啟動通信線程
        self.start_communication_threads()
        
        self.get_logger().info('LLaVA Server Node initialized successfully')
        self.get_logger().info(f'Model loaded: {self.model_worker.model_loaded}')
    
    def setup_ros_interfaces(self):
        """設置 ROS2 介面"""
        # Publishers
        self.grasp_target_pub = self.create_publisher(GraspTarget, '/grasp_target', 10)
        self.annotated_image_pub = self.create_publisher(AnnotatedImage, '/annotated_image', 10)
        
        # Subscribers
        self.vla_command_sub = self.create_subscription(
            VLACommand, '/vla_command', self.vla_command_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
            
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Services
        self.process_service = self.create_service(
            ProcessVLARequest, 'process_vla_request', self.process_vla_request_callback)
    
    def setup_zmq_connections(self):
        """設置 ZMQ 連接"""
        self.zmq_context = zmq.Context()
        
        # 接收命令的 subscriber
        self.zmq_sub = self.zmq_context.socket(zmq.SUB)
        self.zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_sub.setsockopt(zmq.SNDHWM, 1)
        self.zmq_sub.setsockopt(zmq.RCVHWM, 1)
        self.zmq_sub.setsockopt(zmq.CONFLATE, 1)
        
        # 連接到命令來源
        if self.use_remote_computer:
            command_address = f'tcp://{vn.robot_ip}:{vn.nl_command_port}'
        else:
            command_address = f'tcp://127.0.0.1:{vn.nl_command_port}'
        
        self.zmq_sub.connect(command_address)
        self.get_logger().info(f'ZMQ subscriber connected to {command_address}')
        
        # 發布結果的 publisher
        self.zmq_pub = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub.setsockopt(zmq.SNDHWM, 1)
        self.zmq_pub.setsockopt(zmq.RCVHWM, 1)
        
        result_address = f'tcp://*:{vn.nl_target_port}'
        self.zmq_pub.bind(result_address)
        self.get_logger().info(f'ZMQ publisher bound to {result_address}')
        
        time.sleep(0.1)  # 等待連接建立
    
    def start_communication_threads(self):
        """啟動通信線程"""
        self.running = True
        
        # ZMQ 命令處理線程
        self.command_thread = threading.Thread(target=self.command_processing_loop, daemon=True)
        self.command_thread.start()
    
    def depth_callback(self, msg):
        """深度圖像回調"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {str(e)}')
    
    def camera_info_callback(self, msg):
        """相機參數回調"""
        self.camera_info = msg
    
    def vla_command_callback(self, msg):
        """VLA 命令回調"""
        try:
            # 轉換圖像
            cv_image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
            self.latest_cv_image = cv_image
            
            # 執行物件檢測
            self.process_detection_request(cv_image, msg.command_text)
            
        except Exception as e:
            self.get_logger().error(f'Error processing VLA command: {str(e)}')
    
    def process_detection_request(self, image, command):
        """處理檢測請求"""
        try:
            # 使用模型工作器進行檢測
            detection_result = self.model_worker.detect_object(image, command)
            
            # 標注圖像
            annotated_image, grasp_targets = self.annotator.annotate_image(image, detection_result)
            
            # 發布結果
            self.publish_results(annotated_image, grasp_targets, command)
            
            self.get_logger().info(f'Processed command: "{command}", found {len(grasp_targets)} targets')
            
        except Exception as e:
            self.get_logger().error(f'Error processing detection request: {str(e)}')
    
    def publish_results(self, annotated_image, grasp_targets, command):
        """發布檢測結果"""
        # 發布 grasp targets
        for target_data in grasp_targets:
            target_msg = GraspTarget()
            target_msg.target_position_2d.x = target_data['pixel_x']
            target_msg.target_position_2d.y = target_data['pixel_y']
            target_msg.target_position_2d.z = 0.0
            
            # 如果有深度和相機參數，計算 3D 座標
            if self.latest_depth_image is not None and self.camera_info is not None:
                pos_3d = self.pixel_to_3d(
                    int(target_data['pixel_x']), 
                    int(target_data['pixel_y']), 
                    self.latest_depth_image, 
                    self.camera_info
                )
                if pos_3d is not None:
                    target_msg.target_position_3d.x = float(pos_3d[0])
                    target_msg.target_position_3d.y = float(pos_3d[1])
                    target_msg.target_position_3d.z = float(pos_3d[2])
            
            target_msg.confidence = target_data['confidence']
            target_msg.object_description = target_data['description']
            target_msg.timestamp = int(time.time() * 1000)
            
            self.grasp_target_pub.publish(target_msg)
        
        # 發布標注圖像
        if grasp_targets:  # 只有在有檢測結果時才發布
            annotated_msg = AnnotatedImage()
            annotated_msg.image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            
            # 轉換 grasp_targets 為 ROS 消息格式
            ros_targets = []
            for target_data in grasp_targets:
                target_msg = GraspTarget()
                target_msg.target_position_2d.x = target_data['pixel_x']
                target_msg.target_position_2d.y = target_data['pixel_y']
                target_msg.confidence = target_data['confidence']
                target_msg.object_description = target_data['description']
                target_msg.timestamp = int(time.time() * 1000)
                ros_targets.append(target_msg)
            
            annotated_msg.detected_targets = ros_targets
            annotated_msg.timestamp = int(time.time() * 1000)
            
            self.annotated_image_pub.publish(annotated_msg)
    
    def pixel_to_3d(self, pixel_x, pixel_y, depth_image, camera_info, window_size=5):
        """將像素座標轉換為 3D 座標"""
        try:
            h, w = depth_image.shape
            
            # 確保像素在範圍內
            pixel_x = max(0, min(w-1, int(pixel_x)))
            pixel_y = max(0, min(h-1, int(pixel_y)))
            
            # 使用中位數濾波獲取深度值
            half_window = window_size // 2
            x_start = max(0, pixel_x - half_window)
            x_end = min(w, pixel_x + half_window + 1)
            y_start = max(0, pixel_y - half_window)
            y_end = min(h, pixel_y + half_window + 1)
            
            depth_window = depth_image[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[depth_window > 0]
            
            if len(valid_depths) == 0:
                return None
                
            depth_m = float(np.median(valid_depths)) * 0.001  # 轉換為米
            
            if depth_m <= 0:
                return None
            
            # 提取相機內參
            K = np.array(camera_info.k).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # 轉換為 3D 座標
            x = (pixel_x - cx) * depth_m / fx
            y = (pixel_y - cy) * depth_m / fy
            z = depth_m
            
            return np.array([x, y, z])
            
        except Exception as e:
            self.get_logger().error(f'Error in pixel_to_3d: {e}')
            return None
    
    def process_vla_request_callback(self, request, response):
        """服務回調函數"""
        try:
            # 轉換圖像
            cv_image = self.bridge.imgmsg_to_cv2(request.image, "bgr8")
            
            # 執行檢測
            detection_result = self.model_worker.detect_object(cv_image, request.command)
            
            # 設置回應
            if detection_result and detection_result.get('center_px'):
                response.success = True
                
                # 創建 grasp target
                target_msg = GraspTarget()
                center_px = detection_result['center_px']
                target_msg.target_position_2d.x = float(center_px[0])
                target_msg.target_position_2d.y = float(center_px[1])
                target_msg.confidence = detection_result.get('confidence', 0.0)
                target_msg.object_description = detection_result.get('description', 'unknown')
                
                response.grasp_target = target_msg
                response.message = f"Found {target_msg.object_description}"
            else:
                response.success = False
                response.message = "No objects found"
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response
    
    def command_processing_loop(self):
        """ZMQ 命令處理循環"""
        while self.running:
            try:
                if self.zmq_sub.poll(100):  # 100ms 超時
                    message = self.zmq_sub.recv_string(zmq.NOBLOCK)
                    self.get_logger().info(f'Received ZMQ message: {message}')
                    
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        self.get_logger().error(f'Failed to parse JSON: {message}')
                        continue
                    
                    command = data.get('command', '')
                    timestamp = data.get('timestamp', 0)
                    image_data = data.get('image', None)
                    
                    self.get_logger().info(f'Processing command: "{command}", has_image: {image_data is not None}')
                    
                    if command:
                        # 獲取影像
                        current_image = None
                        
                        # 優先使用命令中的影像
                        if image_data:
                            try:
                                # 解碼 base64 影像
                                img_bytes = base64.b64decode(image_data)
                                img_array = np.frombuffer(img_bytes, np.uint8)
                                current_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                self.get_logger().info(f'Decoded image from command: {current_image.shape}')
                            except Exception as e:
                                self.get_logger().error(f'Error decoding image from command: {e}')
                        
                        # 備用：使用最新的 ROS 影像
                        if current_image is None and self.latest_cv_image is not None:
                            current_image = self.latest_cv_image.copy()
                            self.get_logger().info(f'Using latest ROS image: {current_image.shape}')
                        
                        if current_image is not None:
                            # 處理命令
                            detection_result = self.model_worker.detect_object(current_image, command)
                            self.get_logger().info(f'Detection result: {detection_result}')
                            
                            annotated_image, grasp_targets = self.annotator.annotate_image(
                                current_image, detection_result)
                            
                            # 將標註圖像編碼為 base64 用於 ZMQ 傳送
                            _, buffer = cv2.imencode('.jpg', annotated_image)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # 透過 ZMQ 發送結果
                            result = {
                                'command': command,
                                'timestamp': timestamp,
                                'targets_found': len(grasp_targets),
                                'targets': grasp_targets,
                                'annotated_image': img_base64,
                                'image_shape': list(annotated_image.shape)
                            }
                            
                            result_json = json.dumps(result)
                            self.zmq_pub.send_string(result_json)
                            self.get_logger().info(f'Sent ZMQ result: {len(grasp_targets)} targets found')
                            
                            # 也透過 ROS 發布
                            self.publish_results(annotated_image, grasp_targets, command)
                            
                        else:
                            self.get_logger().warning('No image available for processing')
                            # 發送錯誤回應
                            error_result = {
                                'command': command,
                                'timestamp': timestamp,
                                'error': 'No image available',
                                'targets_found': 0,
                                'targets': []
                            }
                            self.zmq_pub.send_string(json.dumps(error_result))
                        
            except zmq.Again:
                pass
            except Exception as e:
                self.get_logger().error(f'Error in command processing loop: {str(e)}')
                import traceback
                self.get_logger().error(f'Traceback: {traceback.format_exc()}')


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='LLaVA Server Node',
        description='LLaVA server with object detection and point annotation using Transformers models.'
    )
    parser.add_argument('-r', '--remote', action='store_true',
                       help='Use remote computer for VLA processing')
    parser.add_argument('--model-type', choices=['transformers', 'ollama'], default='transformers',
                       help='Model backend type')
    parser.add_argument('--model-path', default='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
                       help='Model path or name (HuggingFace model ID or local path)')
    
    # Parse only known args to avoid conflicts with ROS args
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    node = LLaVAServerNode(
        model_type=parsed_args.model_type,
        model_path=parsed_args.model_path,
        use_remote_computer=parsed_args.remote
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLaVA server node')
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
