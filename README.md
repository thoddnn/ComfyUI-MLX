# ComfyUI MLX Nodes

Faster workflows for ComfyUI users on Mac with Apple silicon

## Installation

1. First, install the DiffusionKit library:

```bash
conda create -n comfy_mlx python=3.11 -y
conda activate comfy_mlx
cd /path/to/your_folder
git clone https://github.com/argmaxinc/DiffusionKit
cd DiffusionKit
pip install -e .
```

2. Install the MLX nodes:
    
 - In ComfyUI, Manager > Custom Nodes Manager > Tap 'ComfyUI MLX' > Click Install 

 OR 
 
 - In ComfyUI, Manager > Install via Git URL > https://github.com/thoddnn/ComfyUI-MLX.git

## Getting Started

A basic workflow is provided to help you start experimenting with the nodes.

## Why ComfyUI MLX Nodes?

I started building these nodes because image generation from Flux models was taking too much time on my MacBook. After discovering DiffusionKit on X, which showcased great performance for image generation on Apple Silicon, I decided to create a quick port of the library.

The goal is to collaborate with other contributors to build a full suite of ComfyUI custom nodes optimized for Apple Silicon. 

Additionally, we aim to minimize the reliance on torch to take full advantage of future MLX improvements and further enhance performance.

This will allow ComfyUI users on Mac with Apple Silicon to experience faster workflows.

## Contributing 

Contributions are welcome! I'm open to best practices and suggestions and you’re encouraged to submit a Pull Request to improve the project. 🙏

## Future Plans

- Loading models from local file 
- SDXL models support
- ControlNet support
- LoRA support 
- LLM and VLM nodes
- CogXVideo models support  
- Build more MLX based nodes for common workflows (based on your requests)

## License

ComfyUI MLX Nodes is released under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgements

- [DiffusionKit](https://github.com/argmaxinc/DiffusionKit)

## Support

If you encounter any problems or have any questions, please open an issue in this repository.