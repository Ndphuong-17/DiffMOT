import torch
from torch import nn
from models.components import MLP, TransAoA




class UpTriangle1(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1):
        super(UpTriangle1, self).__init__()
        # Define up, down, and mid layers
        self.up = TransAoA(input_size=in_features, output_size=out_features, num_layers=num_layers)

    def forward(self, input, ctx):
        x_10 = self.up(input, ctx)         # Upscale (batch, out_features)
        return x_10
    
class MidTriangle1(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=3, dropout=0.1):
        super(MidTriangle1, self).__init__()

        self.mid_linear = nn.Linear(in_features, out_features)
        self.mid_norm = nn.LayerNorm(out_features)
        self.mid_activation = nn.GELU()
        self.mid_dropout = nn.Dropout(dropout)

        self.mid_linear1 = nn.Linear(out_features * num_nodes, out_features)
        self.mid_norm1 = nn.LayerNorm(out_features)

        self.mid_attention = nn.MultiheadAttention(embed_dim=out_features, num_heads=4, batch_first=True)

    def forward(self, input_up, input_down: list[torch.Tensor], ctx):
        # Process input_up
        input_up_down = self.mid_linear(input_up)
        input_up_down = self.mid_norm(input_up_down)
        input_up_down = self.mid_activation(input_up_down)
        input_up_down = self.mid_dropout(input_up_down)

        # Concatenate with input_down
        input_down.append(input_up_down)
        x_mid = torch.cat(input_down, dim=1)
        x_mid = self.mid_linear1(x_mid)
        x_mid = self.mid_norm1(x_mid)
        x_mid = self.mid_activation(x_mid)
        x_mid = self.mid_dropout(x_mid)

        # Apply attention
        attn_output, _ = self.mid_attention(
            x_mid.unsqueeze(1), x_mid.unsqueeze(1), x_mid.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)

        # Combine with input_up_down
        output = attn_output #+ input_up_down

        return output
   
class DownTriangle1(nn.Module):
    def __init__(self, out_features, num_layers=1):
        super(DownTriangle1, self).__init__()

        self.final_transform = TransAoA(input_size=out_features, output_size=out_features, num_layers=num_layers)
    def forward(self, mid, ctx):
        # Add final transformation
        output = self.final_transform(mid, ctx)  # Add residual from last downscaled input

        return output


    

class SimpleReUNet2Plus(nn.Module):
  def __init__(self, 
               noise_dim = 4, 
               num_layers = 1, 
               hidden_size = 256, 
               filters = [16, 64, 128, 256, 512, 1024, 2048, 4096], 
               mid = True,
               L = 4,
               deep_supervision=False,
               ):
    super(SimpleReUNet2Plus, self).__init__()
    self.noise_dim = noise_dim
    self.num_layers = num_layers
    self.filters = filters
    self.reversed_filters = filters[::-1]
    self.shared_ctx_mlp = MLP(in_features = hidden_size + 3,
                              out_features = hidden_size)
    self.prediction = MLP(in_features = self.filters[0],
                          out_features = noise_dim)
    
    self.layers = L
    self.deep_supervision = deep_supervision
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  
    ## --- j = 0, UPSAMPLER ---
    self.up_00 = TransAoA(input_size=4, output_size=filters[0], num_layers=num_layers)#MLP(in_features = 4, out_features = filters[0])
    self.up_00_10 = UpTriangle1(in_features=filters[0], out_features = filters[1], num_layers = num_layers)
    self.up_10_20 = UpTriangle1(in_features=filters[1], out_features = filters[2], num_layers = num_layers)
    self.up_20_30 = UpTriangle1(in_features=filters[2], out_features = filters[3], num_layers = num_layers)
    self.up_30_40 = UpTriangle1(in_features=filters[3], out_features = filters[4], num_layers = num_layers)


    ## --- j = 2, DOWNSAMPLER ---
    self.down_01 = MidTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 2)
    self.down_11 = MidTriangle1(in_features=filters[2], out_features = filters[1], num_nodes = 2)
    self.down_21 = MidTriangle1(in_features=filters[3], out_features = filters[2], num_nodes = 2)
    self.down_31 = MidTriangle1(in_features=filters[4], out_features = filters[3], num_nodes = 2)

    self.down_0 = DownTriangle1(out_features = filters[0], num_layers = num_layers)
    self.down_1 = DownTriangle1(out_features = filters[1], num_layers = num_layers)
    self.down_2 = DownTriangle1(out_features = filters[2], num_layers = num_layers)
    self.down_3 = DownTriangle1(out_features = filters[3], num_layers = num_layers)

    ## --- j = 2, DOWNSAMPLER ---
    self.down_02 = MidTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 3)
    self.down_12 = MidTriangle1(in_features=filters[2], out_features = filters[1], num_nodes = 3)
    self.down_22 = MidTriangle1(in_features=filters[3], out_features = filters[2], num_nodes = 3)
    
    ## --- j = 3, DOWNSAMPLER ---
    self.down_03 = MidTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 4)
    self.down_13 = MidTriangle1(in_features=filters[2], out_features = filters[1], num_nodes = 4)

    ## --- j = 4, DOWNSAMPLER ---
    self.down_04 = MidTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 5)


  def forward(self, x, beta, context):
    batch_size = x.size(0)
    beta = beta.view(batch_size, 1) # (B, 1)
    context = context.view(batch_size, -1)   # (B, F)
    time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
    ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1).to(self.device)) # (B, 256)

    if not (1 <= self.layers <= 4):
        raise ValueError("the model pruning factor `L` should be 1 <= L <= 3")


    ## --- L = 1 ---
    x_00 = self.up_00(x, ctx_emb)  # (B, 16)
    x_10 = self.up_00_10(x_00, ctx_emb)  # (B, 64)
    mid = self.down_01(x_10, [x_00], ctx_emb) # (B,16)
    x_01 = self.down_0(mid, ctx_emb) # (B,16)
    output_01 = self.prediction(x_01)

    if self.layers == 1:
      return output_01
    
    ## --- L = 2 ---
    x_20 = self.up_10_20(x_10, ctx_emb)  # (B, 128)
    x_11 = self.down_11(x_20, [x_10], ctx_emb) # (B, 64)
    x_11 = self.down_1(x_11, ctx_emb) # (B, 64)
    x_02 = self.down_02(x_11, [x_00, x_01], ctx_emb) # (B, 16)
    x_02 = self.down_0(x_02, ctx_emb) # (B, 16)
    output_02 = self.prediction(x_02)

    if self.layers == 2 and self.deep_supervision:
      return (output_02 + output_01)/2
    elif self.layers == 2:
      return output_02
    
    
    ## --- L = 3 ---
    x_30 = self.up_20_30(x_20, ctx_emb)  # (B, 256)
    x_21 = self.down_21(x_30, [x_20], ctx_emb) # (B, 128)
    x_21 = self.down_2(x_21, ctx_emb) # (B, 128)
    x_12 = self.down_12(x_21, [x_10, x_11], ctx_emb) # (B, 64)
    x_12 = self.down_1(x_12, ctx_emb) # (B, 64)
    x_03 = self.down_03(x_12, [x_00, x_01, x_02], ctx_emb) # (B, 16)
    x_03 = self.down_0(x_03, ctx_emb) # (B, 16)
    output_03 = self.prediction(x_03)
    
    if self.layers == 3 and self.deep_supervision:
      return (output_03 + output_02 + output_01)/3
    elif self.layers == 3:
      return output_03
    
    # --- L = 4 ---
    x_40 = self.up_30_40(x_30, ctx_emb)  # (B, 512)
    x_31 = self.down_31(x_40, [x_30], ctx_emb)  # (B, 256)
    x_31 = self.down_3(x_31, ctx_emb)  # (B, 256)
    x_22 = self.down_22(x_31, [x_20, x_21], ctx_emb)  # (B, 128)
    x_22 = self.down_2(x_22, ctx_emb)  # (B, 128)
    x_13 = self.down_13(x_22, [x_10, x_11, x_12], ctx_emb)  # (B, 64)
    x_13 = self.down_1(x_13, ctx_emb)  # (B, 64)
    x_04 = self.down_04(x_13, [x_00, x_01, x_02, x_03], ctx_emb)  # (B, 16)
    x_04 = self.down_0(x_04, ctx_emb)  # (B, 16)
    output_04 = self.prediction(x_04)

    if self.layers == 4:
        return (output_04 + output_03 + output_02 + output_01) / 4 if self.deep_supervision else output_04

    



if __name__ == '__main__':
   # Initialize the model
    model = SimpleReUNet2Plus()

    # Dummy inputs
    x = torch.randn(16, 4)  # (batch_size, input_dim)
    beta = torch.rand(16, 1)  # (batch_size, 1)
    context = torch.randn(16, 256)  # (batch_size, hidden_size)

    # Test forward pass
    output = model(x, beta, context)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
