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
    Tokenize prompt with image tokens
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        elif return_tensors == 'np':
            return np.array(input_ids)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    
    return input_ids

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
