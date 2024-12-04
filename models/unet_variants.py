import torch
from torch import nn
from einops import rearrange, repeat
import math
from .components import MLP, TransAoA, ReUnet3PlusDownBlock, ReUnet3PlusDownBlock_Smaller
from .ReUNet2Plus1 import ReUNet2Plus1 as ReUNet2Plus
from .SimpleReUNet2Plus1 import SimpleReUNet2Plus
from .SimpleReUNet2Plus2 import SimpleReUNet2Plus as SimpleReUNet2Plus1

class ReUNet(nn.Module):
  def __init__(self, noise_dim = 4, num_layers = 1, hidden_size = 256, filters = [16, 64, 128, 256], mid = True):
    super(ReUNet, self).__init__()
    self.noise_dim = noise_dim
    self.num_layers = num_layers
    self.filters = filters
    self.reversed_filters = filters[::-1]
    self.mid = mid
    self.shared_ctx_mlp = MLP(in_features = hidden_size + 3,
                              out_features = hidden_size)
    self.up_blocks, self.down_blocks = nn.ModuleList(), nn.ModuleList()
    self.prediction = MLP(in_features = self.filters[0],
                          out_features = noise_dim)

    ## -------------UP--------------
    input_size = noise_dim
    for filter in self.filters:
      block = TransAoA(input_size = input_size,
                        output_size = filter,
                        num_layers = num_layers,)
      self.up_blocks.append(block)
      input_size = filter

    ## -------------DOWN--------------
    for i in range(len(self.reversed_filters) - 1):
      block = TransAoA(input_size = self.reversed_filters[i],
                        output_size = self.reversed_filters[i+1],
                        num_layers = num_layers,)
      self.down_blocks.append(block)

    ## -------------MID--------------

    # if self.mid
    # '''stage 4d'''

  def forward(self, x, beta, context):
    batch_size = x.size(0)
    beta = beta.view(batch_size, 1) # (B, 1)
    context = context.view(batch_size, -1)   # (B, F)
    time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
    ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1)) # (B, 256)

    output = x # 16, 4
    ## -------------UP--------------
    for i, block in enumerate(self.up_blocks):
      output = block(input = output,
                      ctx = ctx_emb) # 16, 4^i
    ## -------------MID--------------

    # if self.mid

    ## -------------DOWN--------------
    for i, block in enumerate(self.down_blocks):
      output = block(input = output,
                      ctx = ctx_emb)

    output = self.prediction(output)
    return output
  
class ReUNet3Plus_Smaller(nn.Module):
  def __init__(self, noise_dim = 4, num_layers = 1, hidden_size = 256, filters = [16, 64, 128, 256], mid = True):
    super(ReUNet3Plus_Smaller, self).__init__()
    self.noise_dim = noise_dim
    self.num_layers = num_layers
    self.filters = filters
    self.reversed_filters = filters[::-1]
    self.shared_ctx_mlp = MLP(in_features = hidden_size + 3,
                              out_features = hidden_size)
    self.up_blocks, self.down_blocks = nn.ModuleList(), nn.ModuleList()
    self.prediction = MLP(in_features = self.filters[0],
                          out_features = noise_dim)
    
    ## -------------UP--------------
    input_size = noise_dim
    for filter in self.filters:
      block = TransAoA(input_size = input_size,
                        output_size = filter,
                        num_layers = num_layers,)
      self.up_blocks.append(block)
      input_size = filter
 
    ## -------------DOWN--------------
    total_num = len(self.reversed_filters)
    for i in range(len(self.reversed_filters) - 1):
      down_num = i+1
      mid_num = 1
      up_num = total_num - down_num - mid_num
      block = ReUnet3PlusDownBlock_Smaller(down_num = down_num,
                                          up_num = up_num,
                                          mid_num = mid_num,
                                          filter = self.filters,
                                          reversed_filters = self.reversed_filters,
                                          num_layers = 1,)
      self.down_blocks.append(block)  

  def forward(self, x, beta, context):
    batch_size = x.size(0)
    beta = beta.view(batch_size, 1) # (B, 1)
    context = context.view(batch_size, -1)   # (B, F)
    time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
    ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1)) # (B, 256)

    output = x # 16, 4
    up_sampling_list, down_sampling_list = [], []
    ## -------------UP--------------
    for i, block in enumerate(self.up_blocks):
      output = block(input = output,
                      ctx = ctx_emb) # 16, 4^i
      up_sampling_list.append(output)

    ## -------------MID--------------
    # if self.mid

    ## -------------DOWN--------------
    down_sampling_list.append(up_sampling_list[-1])
    total_num = len(self.down_blocks) + 1

    for i, block in enumerate(self.down_blocks):
      down_num = i+1
      mid_num = 1
      up_num = total_num - down_num - mid_num

      output = block(down_samples = down_sampling_list[:down_num],
                     up_samples = up_sampling_list[:up_num],
                     mid_samples = up_sampling_list[up_num],
                     ctx = ctx_emb)
      down_sampling_list.append(output)

    output = self.prediction(down_sampling_list[-1])
    return output
  
class ReUNet3Plus(nn.Module):
  def __init__(self, noise_dim = 4, num_layers = 1, hidden_size = 256, filters = [16, 64, 128, 256], mid = True):
    super(ReUNet3Plus, self).__init__()
    self.noise_dim = noise_dim
    self.num_layers = num_layers
    self.filters = filters
    self.reversed_filters = filters[::-1]
    self.shared_ctx_mlp = MLP(in_features = hidden_size + 3,
                              out_features = hidden_size)
    self.up_blocks, self.down_blocks = nn.ModuleList(), nn.ModuleList()
    self.prediction = MLP(in_features = self.filters[0],
                          out_features = noise_dim)
    
    ## -------------UP--------------
    input_size = noise_dim
    for filter in self.filters:
      block = TransAoA(input_size = input_size,
                        output_size = filter,
                        num_layers = num_layers,)
      self.up_blocks.append(block)
      input_size = filter
 
    ## -------------DOWN--------------
    total_num = len(self.reversed_filters)
    for i in range(len(self.reversed_filters) - 1):
      down_num = i+1
      mid_num = 1
      up_num = total_num - down_num - mid_num
      block = ReUnet3PlusDownBlock(down_num = down_num,
                                   up_num = up_num,
                                   mid_num = mid_num,
                                   filter = self.filters,
                                   reversed_filters = self.reversed_filters,
                                   num_layers = 1,)
      self.down_blocks.append(block)  

  def forward(self, x, beta, context):
    batch_size = x.size(0)
    beta = beta.view(batch_size, 1) # (B, 1)
    context = context.view(batch_size, -1)   # (B, F)
    time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
    ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1)) # (B, 256)

    output = x # 16, 4
    up_sampling_list, down_sampling_list = [], []
    ## -------------UP--------------
    for i, block in enumerate(self.up_blocks):
      output = block(input = output,
                      ctx = ctx_emb) # 16, 4^i
      up_sampling_list.append(output)

    ## -------------MID--------------
    # if self.mid

    ## -------------DOWN--------------
    down_sampling_list.append(up_sampling_list[-1])
    total_num = len(self.down_blocks) + 1

    for i, block in enumerate(self.down_blocks):
      down_num = i+1
      mid_num = 1
      up_num = total_num - down_num - mid_num

      output = block(down_samples = down_sampling_list[:down_num],
                     up_samples = up_sampling_list[:up_num],
                     mid_samples = up_sampling_list[up_num],
                     ctx = ctx_emb)
      down_sampling_list.append(output)

    output = self.prediction(down_sampling_list[-1])
    return output
