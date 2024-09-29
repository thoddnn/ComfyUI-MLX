import numpy as np
import PIL.Image
import mlx.core as mx
from typing import Optional, Tuple
from PIL import Image
from diffusionkit.mlx.tokenizer import Tokenizer, T5Tokenizer
from diffusionkit.mlx.t5 import SD3T5Encoder
from diffusionkit.mlx import load_t5_encoder, load_t5_tokenizer, load_tokenizer, load_text_encoder
from diffusionkit.mlx.clip import CLIPTextModel
from diffusionkit.mlx.model_io import load_flux
from diffusionkit.mlx import FluxPipeline
import torch

class MLXDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_image": ("LATENT", ), "vae": ("VAE", )}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    
    def decode(self, latent_image, vae):
        decoded = vae(latent_image)
        decoded = mx.clip(decoded / 2 + 0.5, 0, 1)
        
        mx.eval(decoded)

        # Convert the decoded images to uint8
        #x = mx.concatenate(decoded, axis=0)
        #x = (x * 255).astype(mx.uint8)

        # Convert MLX array to numpy array
        numpy_array = np.array(decoded)

        # Convert numpy array to PyTorch tensor
        torch_tensor = torch.from_numpy(numpy_array)

        # Ensure the tensor is float and in the range [0, 1]
        torch_tensor = torch_tensor.float().clamp(0, 1)

        # If the tensor is not in the format [batch, height, width, channels],
        # you might need to permute it
        if torch_tensor.shape[0] == 3:  # If it's in [channels, height, width] format
            torch_tensor = torch_tensor.permute(1, 2, 0)

        # Add a batch dimension if it's not present
        if len(torch_tensor.shape) == 3:
            torch_tensor = torch_tensor.unsqueeze(0)

        return (torch_tensor,)


class MLXSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {"model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "positive": ("CONDITIONING", ),
            "latent_image": ("LATENT", ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_image"

    def generate_image(self, model, seed, steps, cfg, positive, latent_image, denoise): 
        
        conditioning = positive["conditioning"]
        pooled_conditioning = positive["pooled_conditioning"]
        num_steps = steps 
        cfg_weight = cfg
        
        m:FluxPipeline = model

        batch, channels, height, width = latent_image["samples"].shape
        
        latent_size = (height, width)
        
        latents, iter_time  = m.denoise_latents(
            conditioning,
            pooled_conditioning,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=latent_size,
            seed=seed,
            image_path=None,
            denoise=denoise,
        )

        mx.eval(latents)

        latents = latents.astype(m.activation_dtype)

        return (latents,)


class MLXLoadFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_version": (["argmaxinc/mlx-FLUX.1-schnell", 
                               "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized", 
                               "argmaxinc/mlx-FLUX.1-dev"],)
        }}
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    FUNCTION = "load_flux_model"

    def load_flux_model(self, model_version):
        model = FluxPipeline(model_version=model_version, low_memory_mode=True)

        clip = {
            "clip_l_model": model.clip_l,
            "clip_l_tokenizer": model.tokenizer_l,
            "t5_model": model.t5_encoder,
            "t5_tokenizer": model.t5_tokenizer
        }

        return (model, model.decoder, clip)


class MLXClipTextEncoder: 

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", )}}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
        if negative_text is None:
            negative_text = ""
        if tokenizer.pad_with_eos:
            pad_token = tokenizer.eos_token
        else:
            pad_token = 0

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

    def encode(self, clip, text):

        clip_l_encoder:CLIPTextModel = clip["clip_l_model"]
        clip_l_tokenizer:Tokenizer = clip["clip_l_tokenizer"]
        t5_encoder:SD3T5Encoder = clip["t5_model"]
        t5_tokenizer:T5Tokenizer = clip["t5_tokenizer"]

        # CLIP processing
        clip_tokens = self._tokenize(tokenizer=clip_l_tokenizer, text=text) 

        clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :]) 

        clip_last_hidden_state = clip_l_embeddings.last_hidden_state
        clip_pooled_output = clip_l_embeddings.pooled_output
        
        # T5 processing
        t5_tokens = self._tokenize(tokenizer=t5_tokenizer, text=text) 

        padded_tokens_t5 = mx.zeros((1, 256)).astype(
            t5_tokens.dtype
        )

        padded_tokens_t5[:, : t5_tokens.shape[1]] = t5_tokens[
            [0], :
        ]  # Ignore negative text

        t5_embeddings = t5_encoder(padded_tokens_t5)

        # Use T5 embeddings as main conditioning
        conditioning = t5_embeddings
        
        mx.eval(t5_embeddings)
        mx.eval(clip_pooled_output)

        output = {
            "conditioning": t5_embeddings,
            "pooled_conditioning": clip_pooled_output
        }

        return (output, ) 

class MLXFluxCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_flux_clip"
    
    def load_flux_clip(self):
        
        clip_l_tokenizer = load_tokenizer("argmaxinc/stable-diffusion", vocab_key="tokenizer_l_vocab", merges_key="tokenizer_l_merges")
        clip_l_encoder = load_text_encoder("argmaxinc/stable-diffusion", model_key="clip_l")
        
        t5_tokenizer = load_t5_tokenizer(max_context_length=256)
        t5_encoder = load_t5_encoder()

        output = {
            "clip_l_model": clip_l_encoder,
            "clip_l_tokenizer": clip_l_tokenizer,
            "t5_model": t5_encoder,
            "t5_tokenizer": t5_tokenizer
        }
        
        return (output,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MLXFluxCLIPLoader": MLXFluxCLIPLoader,
    "MLXClipTextEncoder": MLXClipTextEncoder,
    "MLXLoadFlux": MLXLoadFlux,
    "MLXSampler": MLXSampler,
    "MLXDecoder": MLXDecoder
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXFluxCLIPLoader": "MLX Flux CLIP Loader",
    "MLXClipTextEncoder": "MLX CLIP Text Encoder",
    "MLXLoadFlux": "MLX Load Flux Model",
    "MLXSampler": "MLX Sampler",
    "MLXDecoder": "MLX Decoder",
}
