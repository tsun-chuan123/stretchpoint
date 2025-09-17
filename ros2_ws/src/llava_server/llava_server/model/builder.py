#!/usr/bin/env python3
"""
Model builder for LLaVA models
Based on RoboPoint implementation
"""

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

# Try to import LLaVA modules
try:
    # Import from the existing LLaVA path in the container
    import sys
    sys.path.insert(0, '/stretchpoint/ros2_ws/src/llava_server')
    
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
    from llava.model.language_model.llava_mpt import LlavaMptForCausalLM
    LLAVA_AVAILABLE = True
except ImportError as e:
    print(f"LLaVA import failed: {e}")
    LLAVA_AVAILABLE = False

# Constants
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


def load_pretrained_model(model_path, model_base=None, model_name=None, 
                         load_8bit=False, load_4bit=False, device_map="auto", 
                         device="cuda", use_flash_attn=False, **kwargs):
    """
    Load pretrained LLaVA model
    
    Args:
        model_path (str): Path to the model
        model_base (str): Base model path for LoRA models
        model_name (str): Model name
        load_8bit (bool): Load model in 8bit
        load_4bit (bool): Load model in 4bit
        device_map (str): Device mapping
        device (str): Device to load model on
        use_flash_attn (bool): Use flash attention
        **kwargs: Additional arguments
    
    Returns:
        tuple: tokenizer, model, image_processor, context_len
    """
    if not LLAVA_AVAILABLE:
        print("Warning: LLaVA not available, loading simplified model")
        return load_simplified_model(model_path, device, **kwargs)
    
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

    # Determine model name if not provided
    if model_name is None:
        model_name = os.path.basename(model_path.rstrip('/'))

    # Handle LoRA models
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided.')
    
    if 'lora' in model_name.lower() and model_base is not None:
        from llava.model.language_model.llava_llama import LlavaConfig
        lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading LLaVA from base model...')
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        
        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # Try to load from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            except Exception as e:
                print(f"Could not load non_lora_trainables: {e}")
                non_lora_trainables = {}
        
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        try:
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        except ImportError:
            print("PEFT not available, skipping LoRA loading")
    
    elif model_base is not None:
        # Load from base model with mm projector
        print('Loading LLaVA from base model...')
        if 'mpt' in model_name.lower():
            if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                if os.path.isfile(os.path.join(model_base, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

        # Load mm projector weights
        if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
    
    else:
        # Load full model
        print(f'Loading full LLaVA model from {model_path}...')
        if 'mpt' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif 'mistral' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = LlavaMistralForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )

    # Configure special tokens
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Load vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # Get context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_simplified_model(model_path, device="cuda", **kwargs):
    """
    Load simplified model when LLaVA is not available
    """
    print(f"Loading simplified model from {model_path}")
    
    try:
        # Try to load as standard causal LM
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=device,
            **kwargs
        )
        
        # Create dummy image processor
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        context_len = 2048
        
        return tokenizer, model, image_processor, context_len
    
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Return None values to indicate failure
        return None, None, None, 2048


def get_model_name_from_path(model_path):
    """Get model name from path"""
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
