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

IMG_SIZE = (64, 64)
BATCH_SIZE = 6
IMG_CHANNELS = 3

IMG_DIM = IMG_SIZE[0] * IMG_SIZE[1] * IMG_CHANNELS


T = 1000  # total diffusion steps
BETA_START = 1e-4
BETA_END = 0.02


# Linear noise schedule

class LinearScheduleDiffuser(nn.Module):
	def __init__(
		self,
		T=1000,                 # total diffusion steps
		beta_start=BETA_START,  # start of beta range
		beta_end=BETA_END,      # end of beta range
		img_shape=(IMG_CHANNELS, *IMG_SIZE),
	):
		super().__init__()
		self.T = T
		self.img_shape = img_shape
		self.beta = torch.linspace(beta_start, beta_end, T)   # β_t ∈ (0.0001, 0.02)
		self.alpha = 1. - self.beta                           # α_t = 1 - β_t
		self.alpha_bar = torch.cumprod(self.alpha, dim=0)     # \bar{α}_t = product of α_1 to α_t
				
		beta = self.get_betas()
		
		self.register_buffer('beta', beta)
		self.register_buffer('alpha', 1 - beta)
		self.register_buffer('sqrt_beta', torch.sqrt(beta))
		self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
		self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
		self.register_buffer('one_by_sqrt_alpha', 1. / torch.sqrt(self.alpha))
		self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - self.alpha_bar))

		
	def forward(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None) -> torch.Tensor:
		"""Forward diffusion process. q(x_t | x_0)"""
		if eps is None:
			eps = torch.randn_like(x0)

		sample  = self.sqrt_alpha_bar[t] * x0 + self.sqrt_one_minus_alpha_bar[t] * eps

		return sample

	def reverse(self, x, t, predicted_noise):
		beta_t = self.beta[t]
		one_by_sqrt_alpha_t = self.one_by_sqrt_alpha[t]
		sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

		if t > 0:
			z = torch.randn_like(x)
		else:
			z = torch.zeros_like(x)

		return (
			one_by_sqrt_alpha_t
			* (x - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
			+ torch.sqrt(beta_t) * z
		)

	def generate_samples(self, denoiser_model, num_samples, device='cpu'):
		with torch.no_grad():
			x_t = torch.randn((num_samples, IMG_CHANNELS, *IMG_SIZE), device=device)
			for t in reversed(range(T)):
				t_tensor = torch.full((num_samples,), t, device=device).long()
				pred_noise = denoiser_model(x_t, t_tensor)
				x_t = self.reverse(x_t, t_tensor, pred_noise)
			
			return x_t

# model_params:
model_config = {
  "im_channels" : IMG_CHANNELS,
  "im_size" : IMG_SIZE[0],
  "down_channels" : [128, 256, 256, 512],
  "mid_channels" : [512, 512, 256],
  # "down_channels" : [32, 64, 128, 256, 512],
  # "mid_channels" : [512, 512, 256],
  "down_sample" : [True, False, True],
  "time_emb_dim" : 128,
  "num_down_layers" : 5,
  "num_mid_layers" : 3,
  "num_up_layers" : 5,
  "num_heads" : 4,
  "dropout_rate" : 0.1,
  "down_apply_attention" : [False, True, False, True, True],
  "mid_apply_attention" : [False, True, False],
  "up_apply_attention" : [False, True, False, True, False],
}

# Checking dimension consistency
assert model_config["mid_channels"][0] == model_config["down_channels"][-1]
assert model_config["mid_channels"][-1] == model_config["down_channels"][-2]
assert len(model_config["down_apply_attention"]) == model_config["num_down_layers"]
assert len(model_config["mid_apply_attention"]) == model_config["num_mid_layers"]
assert len(model_config["up_apply_attention"]) == model_config["num_up_layers"]

# Training parameters
train_config = {
	"num_epochs": 20,
	"batch_size": 6, # Adjusted for GPU memory
	"num_workers": 4,
	"learning_rate": 2e-4,
}

def get_num_parameter(model):
	# Count the number of parameters
	num_params = sum(p.numel() for p in model.parameters())
	return num_params


def get_time_embedding(time_steps, temb_dim: torch.Tensor, device='cpu') -> torch.Tensor:
	r"""
	Convert time steps tensor into an embedding using the
	sinusoidal time embedding formula
	:param time_steps: 1D tensor of length batch size
	:param temb_dim: Dimension of the embedding
	:return: BxD embedding representation of B time steps
	"""
	assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
	half_dim = temb_dim // 2
	# factor = 10000^(2i/d_model)
	factor = 10000 ** ((torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim)
	)
	
	# pos / factor
	# t B -> B, 1 -> B, temb_dim
	t_emb = time_steps[:, None].repeat(1, half_dim) / factor
	t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
	return t_emb

class AttentionBlock(nn.Module):
	def __init__(self, channels=64, num_heads=4, batch_first=False):
		super().__init__()
		self.channels = channels

		self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
		self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=num_heads, batch_first=batch_first)

	def forward(self, x):
		B, _, H, W = x.shape
		h = self.group_norm(x)
		h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
		h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
		h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
		return x + h


class DownEncoder(nn.Module):
	r"""
	Down conv block with attention.
	Sequence of following block
	1. Resnet block with time embedding
	2. Attention block
	3. Downsample using 2x2 average pooling
	"""
	def __init__(self, in_channels, out_channels, t_emb_dim,
				 down_sample=True, num_heads=4, num_layers=1, dropout_rate=0.1, apply_attention=[False, False, True, False]):
		super().__init__()
		self.num_layers = num_layers
		self.down_sample = down_sample
		
		assert len(apply_attention) == num_layers, "apply_attention must have the same length as num_layers"
		
		self.resnet_conv_first = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, in_channels if i == 0 else out_channels),
					nn.SiLU(),
					nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
							  kernel_size=3, stride=1, padding=1),
				)
				for i in range(num_layers)
			]
		)
		self.t_emb_layers = nn.ModuleList([
			nn.Sequential(
				nn.ReLU(),
				nn.Linear(t_emb_dim, out_channels)
			)
			for _ in range(num_layers)
		])
		self.resnet_conv_second = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, out_channels),
					nn.SiLU(),
					nn.Dropout2d(p=dropout_rate),
					nn.Conv2d(out_channels, out_channels,
							  kernel_size=3, stride=1, padding=1),
				)
				for _ in range(num_layers)
			]
		)
		self.residual_input_conv = nn.ModuleList(
			[
				nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
				for i in range(num_layers)
			]
		)
		self.attention = nn.ModuleList(
			[AttentionBlock(channels=out_channels, num_heads=num_heads, batch_first=True) if apply_attention[i] else nn.Identity()
			 for i in range(num_layers)]
		)
		self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
										  4, 2, 1) if self.down_sample else nn.Identity()

	def forward(self, x, t_emb):
		out = x
		for i in range(self.num_layers):
			
			# Resnet block of Unet
			resnet_input = out
			out = self.resnet_conv_first[i](out)
			out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
			out = self.resnet_conv_second[i](out)
			out = out + self.residual_input_conv[i](resnet_input)
			
			# Attention block of Unet
			out = self.attention[i](out)
			
		out = self.down_sample_conv(out)
		return out

class BottleNeck(nn.Module):
	r"""
	Mid conv block with attention.
	Sequence of following blocks
	1. Resnet block with time embedding
	2. Attention block
	3. Resnet block with time embedding
	"""
	def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1, dropout_rate=0.1, apply_attention=[False, True, False]):
		super().__init__()
		self.num_layers = num_layers
		
		assert len(apply_attention) == num_layers, "apply_attention must have the same length as num_layers"
		
		self.resnet_conv_first = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, in_channels if i == 0 else out_channels),
					nn.SiLU(),
					nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
							  padding=1),
				)
				for i in range(num_layers+1)
			]
		)
		self.t_emb_layers = nn.ModuleList([
			nn.Sequential(
				nn.ReLU(),
				nn.Linear(t_emb_dim, out_channels)
			)
			for _ in range(num_layers + 1)
		])
		self.resnet_conv_second = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, out_channels),
					nn.SiLU(),
					nn.Dropout2d(p=dropout_rate),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
				)
				for _ in range(num_layers+1)
			]
		)
		self.residual_input_conv = nn.ModuleList(
			[
				nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
				for i in range(num_layers+1)
			]
		)
		
		self.attention = nn.ModuleList(
			[AttentionBlock(channels=out_channels, num_heads=num_heads, batch_first=True) if apply_attention[i] else nn.Identity()
			 for i in range(num_layers)]
		)
	
	def forward(self, x, t_emb):
		out = x
		
		# First resnet block
		resnet_input = out
		out = self.resnet_conv_first[0](out)
		out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
		out = self.resnet_conv_second[0](out)
		out = out + self.residual_input_conv[0](resnet_input)
		
		for i in range(self.num_layers):

			# Attention Block
			out = self.attention[i](out)
			
			# Resnet Block
			resnet_input = out
			out = self.resnet_conv_first[i+1](out)
			out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
			out = self.resnet_conv_second[i+1](out)
			out = out + self.residual_input_conv[i+1](resnet_input)
		
		return out


class UpDecoder(nn.Module):
	r"""
	Up conv block with attention.
	Sequence of following blocks
	1. Upsample
	1. Concatenate Down block output
	2. Resnet block with time embedding
	3. Attention Block
	"""
	def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1, dropout_rate=0.1, apply_attention=[False, False, True, False]):
		super().__init__()
		self.num_layers = num_layers
		self.up_sample = up_sample
		
		assert len(apply_attention) == num_layers, "apply_attention must have the same length as num_layers"
		
		
		self.resnet_conv_first = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, in_channels if i == 0 else out_channels),
					nn.SiLU(),
					nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
							  padding=1),
				)
				for i in range(num_layers)
			]
		)
		self.t_emb_layers = nn.ModuleList([
			nn.Sequential(
				nn.ReLU(),
				nn.Linear(t_emb_dim, out_channels)
			)
			for _ in range(num_layers)
		])
		self.resnet_conv_second = nn.ModuleList(
			[
				nn.Sequential(
					nn.GroupNorm(8, out_channels),
					nn.SiLU(),
					nn.Dropout2d(p=dropout_rate),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
				)
				for _ in range(num_layers)
			]
		)
		
		self.attention = nn.ModuleList(
			[AttentionBlock(channels=out_channels, num_heads=num_heads, batch_first=True) if apply_attention[i] else nn.Identity()
			 for i in range(num_layers)]
		)
		
		self.residual_input_conv = nn.ModuleList(
			[
				nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
				for i in range(num_layers)
			]
		)
		self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
												 4, 2, 1) \
			if self.up_sample else nn.Identity()
	
	def forward(self, x, out_down, t_emb):
		x = self.up_sample_conv(x)
		x = torch.cat([x, out_down], dim=1)
		
		out = x
		for i in range(self.num_layers):
			# Resnet block
			resnet_input = out
			out = self.resnet_conv_first[i](out)
			out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
			out = self.resnet_conv_second[i](out)
			out = out + self.residual_input_conv[i](resnet_input)
			
			# Attention block
			out = self.attention[i](out)
		return out


class Unet(nn.Module):
	r"""
	Unet model comprising
	Down blocks, Midblocks and Uplocks
	"""
	def __init__(self, model_config, diffusion_steps=T):
		super().__init__()
		im_channels = model_config['im_channels']
		self.down_channels = model_config['down_channels']
		self.mid_channels = model_config['mid_channels']
		self.t_emb_dim = model_config['time_emb_dim']
		self.down_sample = model_config['down_sample']
		self.num_down_layers = model_config['num_down_layers']
		self.num_mid_layers = model_config['num_mid_layers']
		self.num_up_layers = model_config['num_up_layers']
		self.num_heads = model_config['num_heads']
		self.dropout_rate = model_config['dropout_rate']
		self.diffusion_steps = diffusion_steps
		
		assert self.mid_channels[0] == self.down_channels[-1]
		assert self.mid_channels[-1] == self.down_channels[-2]
		assert len(self.down_sample) == len(self.down_channels) - 1

		# Initial projection from sinusoidal time embedding
		self.t_proj = nn.Sequential(
			nn.Linear(self.t_emb_dim, self.t_emb_dim),
			nn.ReLU(),
			nn.Linear(self.t_emb_dim, self.t_emb_dim)
		)

		self.up_sample = list(reversed(self.down_sample))
		self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

		self.downs = nn.ModuleList([])
		for i in range(len(self.down_channels)-1):
			self.downs.append(DownEncoder(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
										down_sample=self.down_sample[i], num_heads=self.num_heads, num_layers=self.num_down_layers, dropout_rate=self.dropout_rate,
										apply_attention=model_config['down_apply_attention']))
		
		self.mids = nn.ModuleList([])
		for i in range(len(self.mid_channels)-1):
			self.mids.append(BottleNeck(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
									  num_heads=self.num_heads, num_layers=self.num_mid_layers, dropout_rate=self.dropout_rate,
									  apply_attention=model_config['mid_apply_attention']))
		
		self.ups = nn.ModuleList([])
		for i in reversed(range(len(self.down_channels)-1)):
			self.ups.append(UpDecoder(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
									self.t_emb_dim, up_sample=self.down_sample[i], num_heads=self.num_heads, num_layers=self.num_up_layers, dropout_rate=self.dropout_rate,
									apply_attention=model_config['up_apply_attention']))
		
		self.norm_out = nn.GroupNorm(8, 16)
		self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
	
	def forward(self, x, t):
		# Shapes assuming downblocks are [C1, C2, C3, C4]
		# Shapes assuming midblocks are [C4, C4, C3]
		# Shapes assuming downsamples are [True, True, False]
		# B x C x H x W
		out = self.conv_in(x)
		# B x C1 x H x W
		
		# t_emb -> B x t_emb_dim
		t_emb = get_time_embedding(torch.as_tensor(t, device=x.device).long(), self.t_emb_dim).cuda()
		t_emb = self.t_proj(t_emb)
		
		down_outs = []
		
		for idx, down in enumerate(self.downs):
			down_outs.append(out)
			out = down(out, t_emb)
		# down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
		# out B x C4 x H/4 x W/4
		
		for mid in self.mids:
			out = mid(out, t_emb)
		# out B x C3 x H/4 x W/4
		
		for up in self.ups:
			down_out = down_outs.pop()
			out = up(out, down_out, t_emb)
			# out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
		out = self.norm_out(out)
		out = nn.SiLU()(out)
		out = self.conv_out(out)
		# out B x C x H x W
		return out
	
	
	# def q_sample(self, x0, t, noise=None): # add noise | forward pass | q(x_t | x_t-1)
	# 	if noise is None:
	# 		noise = torch.randn_like(x0).to(x0.device)

	# 	sqrt_alpha_bar = alpha_bar[t].sqrt().view(-1, 1, 1, 1)
	# 	sqrt_one_minus_alpha_bar = (1 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
		
	# 	return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise # <= x_t 

	def get_loss(self, x0, t=None):
		B = x0.shape[0]
		if t is None:
			t = torch.randint(0, T, (B,), device=x0.device).long()
		noise = torch.randn_like(x0)
		x_t = self.q_sample(x0, t, noise)
		pred_noise = self(x_t, t)
		return nn.MSELoss()(pred_noise, noise), pred_noise, x_t
	
	# def generate_samples(self, num_samples, device='cpu'):
	# 	with torch.no_grad():
	# 		x_t = torch.randn((num_samples, IMG_CHANNELS, *IMG_SIZE), device=device)
	# 		for t in reversed(range(T)):
	# 			t_tensor = torch.full((num_samples,), t, device=device).long()
	# 			pred_noise = self(x_t, t_tensor)
				
	# 			sqrt_alpha_bar_t = alpha_bar[t].sqrt().view(-1, 1, 1, 1)
	# 			sqrt_one_minus_alpha_bar_t = (1 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
				
	# 			# Compute predicted x_0
	# 			pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
				
	# 			if t > 0:
	# 				noise = torch.randn_like(x_t)
	# 				x_t = sqrt_alpha_bar_t * pred_x0 + sqrt_one_minus_alpha_bar_t * noise
	# 			else:
	# 				x_t = pred_x0
			
	# 		return x_t
	
	def get_num_trainable_parameters(self):
		# Count the number of parameters
		num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return num_params
