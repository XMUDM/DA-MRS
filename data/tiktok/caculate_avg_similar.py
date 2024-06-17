import numpy as np
import pandas as pd
import math

import torch

item_text_feat = np.load('text_feat.npy')
item_image_feat = np.load('image_feat.npy')
item_audio_feat = np.load('audio_feat.npy')

item_text_tensor = torch.tensor(item_text_feat).float()
item_image_tensor = torch.tensor(item_image_feat).float()
item_audio_tensor = torch.tensor(item_audio_feat).float()

v_context_norm = item_image_tensor.div(torch.norm(item_image_tensor, p=2, dim=-1, keepdim=True)) # L2 归一化
v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

t_context_norm = item_text_tensor.div(torch.norm(item_text_tensor, p=2, dim=-1, keepdim=True))
t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

a_context_norm = item_audio_tensor.div(torch.norm(item_audio_tensor, p=2, dim=-1, keepdim=True)) # L2 归一化
a_sim = torch.mm(a_context_norm, a_context_norm.transpose(1, 0))

print('V')
print(v_sim.mean())
print('T')
print(t_sim.mean())
print('A')
print(a_sim.mean())




