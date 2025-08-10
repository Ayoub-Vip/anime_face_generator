# anime face generator

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

diffusion model to generate anime faces.

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

