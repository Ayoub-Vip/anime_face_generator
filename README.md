# Anime Face Generator using Diffusion Model
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
  <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
![PIL](https://img.shields.io/badge/PIL-Image%20Processing-green?logo=python&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-fcc200?logo=wandb&logoColor=black)
![SLURM](https://img.shields.io/badge/SLURM-Cluster%20Scheduler-blue?logo=slurm&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)

<!-- <hr/> -->


<div align="center">
  <img src="reports/figures/original_data_samples_grid.png" alt="Original Data Samples" width="60%"/>
</div>

For this project we build a generative deep learning system that can generate anime faces. The system will be trained on a dataset of anime faces and will use denoising diffusion probabilistic models (DDPM) to generate new faces. The goal is to create a system that can generate acceptable quality anime faces that are diverse and realistic by drawing samples, and naturally for denoising by encoding-decoding the image. The project will also explore the use of different architectures and the impact of self-attention on standard DDPM to improve the quality of the generated faces.

Through this project, we:
- Collected and preprocessed over 80,000 anime face images, applying normalization, resizing, and a few augmentations.
- Designed and trained two distinct deep learning models, mainly noise prediction technique and a quick view on score-based energy model, experimenting with different architectures and training strategies.

To see the "conference style" report click on the following GitHub link:  
<a href="https://github.com/Ayoub-Vip/anime_face_generator/blob/master/reports/report.pdf">
 https://github.com/Ayoub-Vip/anime_face_generator/blob/master/reports/report.pdf
</a>

## Some Generated Images

Here are the results of Giga model (85M parameters) for epochs 1, 4, 5, 7, 9, 10, 13, and 16:

<div align="center">
  <img src="reports/figures/85M_params_GIGA_DDPM_Unet_ckpt_epoch_1_epoch_1_samples.png" alt="Epoch 1" width="60%"/>
  <img src="reports/figures/85M_params_GIGA_DDPM_Unet_ckpt_epoch_4_epoch_4_samples.png" alt="Epoch 4" width="60%"/>
  <img src="reports/figures/85M_params_GIGA_DDPM_Unet_ckpt_epoch_5_with_16_samples.png" alt="Epoch 5" width="60%"/>
  <img src="reports/figures/85M_params_GIGA_DDPM_Unet_ckpt_epoch_7_epoch_7_samples.png" alt="Epoch 7" width="60%"/>
  <img src="reports/figures/85M_params_GIGA_DDPM_Unet_ckpt_epoch_9.png" alt="Epoch 9" width="60%"/>
  <img src="reports/figures/giga_unet_ddpm_85M_ckpt_epoch_10_epoch_10_samples.png" alt="Epoch 10" width="60%"/>
  <img src="reports/figures/giga_unet_ddpm_85M_ckpt_epoch_13_epoch_13_samples.png" alt="Epoch 13" width="60%"/>
  <img src="reports/figures/giga_unet_ddpm_85M_ckpt_epoch_16_epoch_16_samples.png" alt="Epoch 16" width="60%"/>
</div>

## Model Architecture

<div align="center">
  <img src="reports/figures/block_layer.png" alt="Model Architecture" width="60%"/>
</div>


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         anime_face_generator and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── anime_face_generator   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes anime_face_generator a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── modeling                <- Contrains models, training and tuning code
    |   |
    │   ├── __init__.py 
    │   ├── generate.py          <- Code for denoising or generating images with trained models          
    │   ├── train.py            <- Code to train models
    │   │
    |   └── models              <- Contains models
    |       ├── 10M-Dummy_Unet_DDPM
    |       ├── 48M_Simple_Unet_DDPM
    |       ├── 75M_Mega_Unet_DDPM
    |       ├── 85M_Giga_Unet_edited_DDPM
    |       └── 85M_Giga_Unet_Score_Based_EDM
    │
    └── plots.py                <- Code to create visualizations
```

--------

