#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import io

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as tdata
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm.notebook import tqdm
import wandb


# In[2]:


from anime_face_generator.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR
import anime_face_generator.modeling.models.mega_unet_ddpm_74M as Mega
from anime_face_generator.modeling.train_ddp import train_model_ddp as train_model

# ![Algorithm1](https://learnopencv.com/wp-content/uploads/2023/02/denoising-diffusion-probabilistic-models_DDPM_trainig_inference_algorithm-1024x247.png)

# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("number of GPUs:", torch.cuda.device_count())


# ## Data augmentation

# In[4]:


transformers = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=Mega.ModelParams.im_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = torchvision.datasets.ImageFolder(root=PROCESSED_DATA_DIR, transform=transformers, target_transform=None)


# In[5]:


rand_gen = torch.Generator().manual_seed(1978)


# ## Unet Denoiser Architecture

# ![image.png](attachment:d0e0108a-f032-4abb-b5c5-1e8804476fd4.png)

# In[9]:

def main():
    mega_model_trained = train_model(module=Mega, dataset=dataset, device=device, rand_gen=rand_gen, wandb_logs=True)



os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
if __name__ == "__main__":
    
    main()