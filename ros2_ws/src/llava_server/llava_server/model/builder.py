#!/usr/bin/env python3
"""
Simplified model builder for LLaVA server using standard Transformers
"""

import os
import warnings
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    BitsAndBytesConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor
)

# Constants - simplified versions
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_TOKEN_INDEX = -200

def load_pretrained_model(model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    """
    Simplified model loading for standard multimodal transformers models
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    print(f"Loading model from: {model_path}")
    
    # Detect model type from config
    try:
        config = AutoConfig.from_pretrained(model_path)
        model_type_name = config.model_type
        print(f"Detected model type: {model_type_name}")
    except:
        model_type_name = "unknown"
    
    # Try to load with appropriate model class based on type
    try:
        if model_type_name == "llava" or "llava" in model_path.lower():
            # Use LLaVA specific classes
            print("Loading with LlavaForConditionalGeneration...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            processor = LlavaProcessor.from_pretrained(model_path)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
            image_processor = processor.image_processor
            
        elif model_type_name == "llava_llama":
            # This is a custom robopoint-style model, will fail gracefully
            print("Detected custom llava_llama model, attempting standard loading...")
            raise ValueError(f"Custom model type {model_type_name} requires specialized loading")
            
        else:
            # Try standard AutoProcessor first
            print("Loading with AutoProcessor and AutoModelForCausalLM...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
            
            # For models that have built-in vision processing
            image_processor = getattr(processor, 'image_processor', processor)
            
        print(f"Loaded model with specialized loading: {type(model)}")
        
    except Exception as e1:
        print(f"Specialized loading failed: {e1}")
        # Fallback to basic loading
        try:
            print("Trying fallback loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
            image_processor = None
            print(f"Loaded model with fallback: {type(model)}")
            
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise e2

    # Add special tokens if needed
    special_tokens = []
    if hasattr(model.config, "mm_use_im_patch_token") and model.config.mm_use_im_patch_token:
        special_tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
    if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    # Try to load vision tower if available
    if hasattr(model, 'get_vision_tower'):
        try:
            vision_tower = model.get_vision_tower()
            if vision_tower and not getattr(vision_tower, 'is_loaded', True):
                vision_tower.load_model(device_map=device_map)
            if vision_tower and device_map != 'auto':
                vision_tower.to(device=device_map, dtype=torch.float16)
            if vision_tower and hasattr(vision_tower, 'image_processor'):
                image_processor = vision_tower.image_processor
        except Exception as e:
            print(f"Vision tower loading failed: {e}")

    # Get context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    else:
        context_len = 2048

    print(f"Model loaded successfully. Context length: {context_len}")
    
    # Return processor if available for LLaVA models
    if 'processor' in locals():
        return tokenizer, model, image_processor, context_len, processor
    else:
        return tokenizer, model, image_processor, context_len
