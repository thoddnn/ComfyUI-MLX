import numpy as np
import PIL.Image
import mlx.core as mx
from typing import Optional, Tuple
from PIL import Image

from .diffusionkit.mlx.tokenizer import Tokenizer, T5Tokenizer
from .diffusionkit.mlx.t5 import SD3T5Encoder
from .diffusionkit.mlx import load_t5_encoder, load_t5_tokenizer, load_tokenizer, load_text_encoder
from .diffusionkit.mlx.clip import CLIPTextModel
from .diffusionkit.mlx.model_io import load_flux
from .diffusionkit.mlx import FluxPipeline
import folder_paths
import torch
import os 
import gc

class MLXDecoder:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_image": ("LATENT", ), "mlx_vae": ("mlx_vae", )}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    
    def decode(self, latent_image, mlx_vae):

        decoded = mlx_vae(latent_image)
        decoded = mx.clip(decoded / 2 + 0.5, 0, 1)

        mx.eval(decoded)
 
        # Convert MLX tensor to numpy array
        decoded_np = np.array(decoded.astype(mx.float16))

        # Convert numpy array to PyTorch tensor
        decoded_torch = torch.from_numpy(decoded_np).float()

        # Ensure the tensor is in the correct format (B, C, H, W)
        if decoded_torch.dim() == 3:
            decoded_torch = decoded_torch.unsqueeze(0)
        
        # Ensure the values are in the range [0, 1]
        decoded_torch = torch.clamp(decoded_torch, 0, 1)

        return (decoded_torch,)


class MLXSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {"mlx_model": ("mlx_model",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "mlx_positive_conditioning": ("mlx_conditioning", ),
            "latent_image": ("LATENT", ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_image"

    def generate_image(self, mlx_model, seed, steps, cfg, mlx_positive_conditioning, latent_image, denoise): 
        
        conditioning = mlx_positive_conditioning["conditioning"]
        pooled_conditioning = mlx_positive_conditioning["pooled_conditioning"]
        num_steps = steps 
        cfg_weight = cfg
            
        batch, channels, height, width = latent_image["samples"].shape
        
        latent_size = (height, width)
        
        latents, iter_time  = mlx_model.denoise_latents(
            conditioning,
            pooled_conditioning,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=latent_size,
            seed=seed,
            image_path=None,
            denoise=denoise,
        )

        latents = latents.astype(mlx_model.activation_dtype)

        return (latents,)


class MLXLoadFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_version": ([
                        "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
                        "argmaxinc/mlx-FLUX.1-schnell",  
                        "argmaxinc/mlx-FLUX.1-dev"
                        ],)
        }}
    
    RETURN_TYPES = ("mlx_model", "mlx_vae", "mlx_conditioning")
    FUNCTION = "load_flux_model"

    def load_flux_model(self, model_version):

        model = FluxPipeline(model_version=model_version, low_memory_mode=True, w16=True, a16=True)

        clip = {
            "model_name": model_version,
            "clip_l_model": model.clip_l,
            "clip_l_tokenizer": model.tokenizer_l,
            "t5_model": model.t5_encoder,
            "t5_tokenizer": model.t5_tokenizer
        }
        
        return (model, model.decoder, clip)


class MLXClipTextEncoder: 

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True}), "mlx_conditioning": ("mlx_conditioning", {"forceInput":True})}}
    

    RETURN_TYPES = ("mlx_conditioning",)
    FUNCTION = "encode"


    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
        if negative_text is None:
            negative_text = ""
        if tokenizer.pad_with_eos:
            pad_token = tokenizer.eos_token
        else:
            pad_token = 0

        text = text.replace('’', '\'')

        # Tokenize the text
        tokens = [tokenizer.tokenize(text)]
        if tokenizer.pad_to_max_length:
            tokens[0].extend([pad_token] * (tokenizer.max_length - len(tokens[0])))
        if negative_text is not None:
            tokens += [tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [pad_token] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        return tokens

    def encode(self, mlx_conditioning, text):

        T5_MAX_LENGTH = {
            "argmaxinc/mlx-stable-diffusion-3-medium": 512,
            "argmaxinc/mlx-FLUX.1-schnell": 256,
            "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 256,
            "argmaxinc/mlx-FLUX.1-dev": 512,
        }

        model_name = mlx_conditioning["model_name"]
        clip_l_encoder:CLIPTextModel = mlx_conditioning["clip_l_model"]
        clip_l_tokenizer:Tokenizer = mlx_conditioning["clip_l_tokenizer"]
        t5_encoder:SD3T5Encoder = mlx_conditioning["t5_model"]
        t5_tokenizer:T5Tokenizer = mlx_conditioning["t5_tokenizer"]

        # CLIP processing
        clip_tokens = self._tokenize(tokenizer=clip_l_tokenizer, text=text) 

        clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :]) 

        clip_last_hidden_state = clip_l_embeddings.last_hidden_state
        clip_pooled_output = clip_l_embeddings.pooled_output
        
        # T5 processing
        t5_tokens = self._tokenize(tokenizer=t5_tokenizer, text=text) 

        padded_tokens_t5 = mx.zeros((1, T5_MAX_LENGTH[model_name])).astype(
            t5_tokens.dtype
        )

        padded_tokens_t5[:, : t5_tokens.shape[1]] = t5_tokens[
            [0], :
        ]  # Ignore negative text

        t5_embeddings = t5_encoder(padded_tokens_t5)

        # Use T5 embeddings as main conditioning
        conditioning = t5_embeddings
        
        output = {
            "conditioning": t5_embeddings,
            "pooled_conditioning": clip_pooled_output
        }

        return (output, ) 

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MLXClipTextEncoder": MLXClipTextEncoder,
    "MLXLoadFlux": MLXLoadFlux,
    "MLXSampler": MLXSampler,
    "MLXDecoder": MLXDecoder
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXClipTextEncoder": "MLX CLIP Text Encoder",
    "MLXLoadFlux": "MLX Load Flux Model from HF 🤗",
    "MLXSampler": "MLX Sampler",
    "MLXDecoder": "MLX Decoder"
}
