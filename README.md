# anime face generator

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

diffusion model to generate anime faces.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, 80K images canonical data sets for modeling.
│   └── raw            <- The original, 63K + 21K images (immutable data dump).
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks, visualization, architecture exploration, contains
│                            Score-Based model basecode in 2.00-VDM-Score-Energy-based.ipynb.
│
├── reports            <- Contains 'report.pdf', and latex code.
│   └── figures        <- Generated figures and plots.
│
├── main_10M_DDPM.ipynb     <- Interface notebook for Mini model, training and sampling.
├── main_49M_DDPM.ipynb     <- Interface notebook for Simple model, training and sampling.
├── main_74M_DDPM.ipynb     <- Interface notebook for Mega model, training and sampling.
├── main_85M_DDPM.ipynb     <- Interface notebook for Giga model, training and sampling.
│
└── anime_face_generator   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes anime_face_generator a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    └── modeling                <- Contrains models, training and tuning code
        |
        ├── __init__.py 
        ├── train.py            <- Code to train models
        ├── train_ddp.py            <- Code to train models using distributed data parallelization
        │
        └── models              <- Contains DDPM model architectures.
            ├── 10M-Dummy_Unet_DDPM
            ├── 48M_Simple_Unet_DDPM
            ├── 75M_Mega_Unet_DDPM
            └── 85M_Giga_Unet_edited_DDPM

```

--------

