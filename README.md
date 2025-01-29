# In-N-Out: Lifting 2D Diffusion Prior for 3D Object Removal via Tuning-Free Latents Alignment
<h3 align="center">NeurIPS, 2024
<h3 align="center"><a href="https://openreview.net/pdf?id=gffaYDu9mM">Paper</a> | <a href="https://timmy11hu.github.io/3dor.github.io/">Project Page</a>


## Installation

To get started, clone this project,
create a conda virtual environment using following <a href="https://docs.nerf.studio/quickstart/installation.html">nerfstudio</a>,
and install the requirements:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
```

## Training
The optimization pipeline is designed to run in a single pass. 
It begins with a 5K-iteration pretraining phase, 
followed by base inpainting and NeRF fine-tuning for an additional 1K iterations. 
After that, the remaining images are inpainted, and joint optimization continues until 10K iterations. 
Videos and inference images are generated every 1K iterations. 
For more details, please refer to the configuration file: ```./nerfacto_v/nerf_config.py```.
```bash
export DATAPATH="./data/$scene_name"
python run_nerf.py --data $DATAPATH
```

## BibTeX

```bibtex
@inproceedings{hu2024innout,
    title        = {In-N-Out: Lifting 2D Diffusion Prior for 3D Object Removal via Tuning-Free Latents Alignment},
    author       = {Dongting Hu and Huan Fu and Jiaxian Guo and Liuhua Peng and Tingjin Chu and Feng Liu and Tongliang Liu and Mingming Gong},
    booktitle    = {The Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    url          = {https://openreview.net/forum?id=gffaYDu9mM}
}
```

## Acknowledgements

The project is based on [NerfStudio](https://docs.nerf.studio/), [Stable Diffusion](https://github.com/Stability-AI/generative-models), [Depth Anything](https://github.com/LiheYoung/Depth-Anything) and [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main). Many thanks to these projects for their excellent contributions!