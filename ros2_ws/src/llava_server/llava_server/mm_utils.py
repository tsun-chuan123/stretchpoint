#!/usr/bin/env python3
"""
Multimodal utilities for LLaVA server
Simplified version of robopoint_worker mm_utils
"""

import torch
import numpy as np
import base64
import io
from PIL import Image
from typing import List, Union
from transformers import AutoImageProcessor, CLIPImageProcessor

# Constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def load_image_from_base64(image_data):
    """Load PIL Image from base64 string"""
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        print(f"Error loading image from base64: {e}")
        return None

def process_images(images, image_processor, model_config):
    """
    Process images for model input
    Simplified version that works with standard transformers
    """
    if image_processor is None:
        # Return dummy tensor if no processor
        return torch.zeros(1, 3, 224, 224)
    
    processed_images = []
    
    for image in images:
        if isinstance(image, str):
            # If base64 string
            image = load_image_from_base64(image)
        
        if image is None:
            continue
            
        # Convert to RGB if needed
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Use the image processor
            if hasattr(image_processor, 'preprocess'):
                processed = image_processor.preprocess(image, return_tensors="pt")
                if 'pixel_values' in processed:
                    processed_images.append(processed['pixel_values'])
                else:
                    processed_images.append(processed)
            elif hasattr(image_processor, '__call__'):
                processed = image_processor(images=image, return_tensors="pt")
                if 'pixel_values' in processed:
                    processed_images.append(processed['pixel_values'])
                else:
                    processed_images.append(processed)
            elif callable(image_processor):
                processed = image_processor(image, return_tensors="pt")
                if 'pixel_values' in processed:
                    processed_images.append(processed['pixel_values'])
                else:
                    processed_images.append(processed)
            else:
                # Fallback: manual preprocessing
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_array = image_array.transpose(2, 0, 1)  # HWC to CHW
                image_tensor = torch.from_numpy(image_array).float() / 255.0
                processed_images.append(image_tensor.unsqueeze(0))
                
        except Exception as e:
            print(f"Error processing image: {e}")
            # Fallback to dummy tensor
            processed_images.append(torch.zeros(1, 3, 224, 224))
    
    if not processed_images:
        return torch.zeros(1, 3, 224, 224)
    
    # For LLaVA models, return the pixel_values directly
    if len(processed_images) == 1:
        return processed_images[0]
    
    # Stack all processed images
    try:
        return torch.cat(processed_images, dim=0)
    except:
        # If stacking fails, return first image
        return processed_images[0] if processed_images else torch.zeros(1, 3, 224, 224)

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenize prompt with image tokens - improved version
    """
    if prompt is None or tokenizer is None:
        print("Warning: prompt or tokenizer is None")
        fallback_ids = [1]  # Use ID 1 as fallback
        if return_tensors == 'pt':
            return torch.tensor(fallback_ids, dtype=torch.long)
        return fallback_ids
    
    try:
        # 更安全的 tokenization 方法
        if '<image>' not in prompt:
            # 沒有圖像標記，直接 tokenize
            tokens = tokenizer(prompt, return_tensors=return_tensors, add_special_tokens=True)
            if return_tensors == 'pt':
                return tokens.input_ids[0]
            return tokens.input_ids
        
        # 分割 prompt
        prompt_chunks = prompt.split('<image>')
        
        # Tokenize 每個部分
        input_ids = []
        
        for i, chunk in enumerate(prompt_chunks):
            if chunk.strip():  # 非空塊
                chunk_tokens = tokenizer(chunk, add_special_tokens=(i == 0))
                if hasattr(chunk_tokens, 'input_ids'):
                    chunk_ids = chunk_tokens.input_ids
                else:
                    chunk_ids = chunk_tokens
                
                # 確保是列表
                if not isinstance(chunk_ids, list):
                    chunk_ids = chunk_ids.tolist() if hasattr(chunk_ids, 'tolist') else [chunk_ids]
                
                # 過濾 None 值
                chunk_ids = [id for id in chunk_ids if id is not None and isinstance(id, int)]
                input_ids.extend(chunk_ids)
            
            # 在塊之間添加圖像標記（除了最後一塊）
            if i < len(prompt_chunks) - 1:
                input_ids.append(image_token_index)
        
        # 確保 input_ids 不為空
        if not input_ids:
            print("Warning: Empty input_ids after tokenization")
            input_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1]
        
        # 驗證所有 ID 都是有效的整數
        valid_ids = []
        for id in input_ids:
            if isinstance(id, int):
                valid_ids.append(id)
            else:
                print(f"Warning: Invalid token ID {id}, replacing with UNK")
                valid_ids.append(tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1)
        
        input_ids = valid_ids
        
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            elif return_tensors == 'np':
                return np.array(input_ids)
            else:
                raise ValueError(f"Unsupported tensor type: {return_tensors}")
        
        return input_ids
        
    except Exception as e:
        print(f"Error in tokenizer_image_token: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a robust fallback token sequence
        try:
            fallback_prompt = prompt.replace('<image>', '[IMG]') if prompt else "Hello"
            fallback_tokens = tokenizer(fallback_prompt, add_special_tokens=True)
            if hasattr(fallback_tokens, 'input_ids'):
                fallback_ids = fallback_tokens.input_ids
                if not isinstance(fallback_ids, list):
                    fallback_ids = fallback_ids.tolist()
            else:
                fallback_ids = [1]
        except:
            fallback_ids = [1]
        
        if return_tensors == 'pt':
            return torch.tensor(fallback_ids, dtype=torch.long)
        return fallback_ids

def expand2square(pil_img, background_color):
    """Expand image to square"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
