import torch
from torch import nn
from models.components import MLP, TransAoA

class UpTriangle1(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, dropout=0.1):
        super(UpTriangle1, self).__init__()

        # Define up, down, and mid layers
        self.up = TransAoA(input_size=in_features, output_size=out_features, num_layers=num_layers)
        self.down = MLP(in_features=out_features, out_features=in_features)
        # self.down = TransAoA(input_size=out_features, output_size=in_features, num_layers=num_layers)
        self.mid = nn.Sequential(
            nn.Linear(in_features * 2, in_features),  # Reduce concatenation size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(in_features),
        )
        self.final_transform = TransAoA(input_size=in_features, output_size=in_features, num_layers=num_layers)

    def forward(self, input, ctx):
        x_00 = input                      # Input tensor (batch, in_features)
        x_10 = self.up(x_00, ctx)         # Upscale (batch, out_features)
        x_10_down = self.down(x_10)  # Downscale (batch, in_features)

        # Concatenate and transform
        x_mid = torch.cat([x_00, x_10_down], dim=1)  # Concatenate along feature dimension
        x_01 = self.mid(x_mid)                      # (batch, in_features)

        # Add skip connection
        x_01 = self.final_transform(x_01 + x_00, ctx)  # Add residual and process

        return x_10, x_01


class DownTriangle1(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, num_nodes=3, dropout=0.1):
        super(DownTriangle1, self).__init__()

        # Downscaling layer
        self.down = MLP(in_features=in_features, out_features=out_features)

        # Mid processing layer
        self.mid = nn.Sequential(
            nn.Linear(out_features * num_nodes, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_features),
        )
        self.final_transform = TransAoA(input_size=out_features, output_size=out_features, num_layers=num_layers)

    def forward(self, input_up, input_down: list[torch.Tensor], ctx):
        # Downscale input_up (batch, out_features)
        input_up_down = self.down(input_up)

        # Append downscaled input to the list
        input_down.append(input_up_down)

        # Concatenate all inputs and process
        concatenated = torch.cat(input_down, dim=1)  # Concatenate list of tensors along feature dimension
        output = self.mid(concatenated)             # Aggregate inputs (batch, out_features)

        # Add final transformation
        output = self.final_transform(output + input_up_down, ctx)  # Add residual from last downscaled input

        return output


    
class SimpleReUNet2Plus(nn.Module):
  def __init__(self, 
               noise_dim = 4, 
               num_layers = 1, 
               hidden_size = 256, 
               filters = [16, 64, 128, 256], 
               mid = True,
               L = 3,
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
    self.up_00 = TransAoA(input_size = 4, output_size = filters[0], num_layers = num_layers)
    self.up_00_10 = UpTriangle1(in_features=filters[0], out_features = filters[1], num_layers = num_layers)
    self.up_10_20 = UpTriangle1(in_features=filters[1], out_features = filters[2], num_layers = num_layers)
    self.up_20_30 = UpTriangle1(in_features=filters[2], out_features = filters[3], num_layers = num_layers)

    ## --- j = 2, DOWNSAMPLER ---
    self.down_02 = DownTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 3, num_layers = num_layers)
    self.down_12 = DownTriangle1(in_features=filters[2], out_features = filters[1], num_nodes = 3, num_layers = num_layers)
    
    ## --- j = 3, DOWNSAMPLER ---
    self.down_03 = DownTriangle1(in_features=filters[1], out_features = filters[0], num_nodes = 4, num_layers = num_layers)



  def forward(self, x, beta, context):
    batch_size = x.size(0)
    beta = beta.view(batch_size, 1) # (B, 1)
    context = context.view(batch_size, -1)   # (B, F)
    time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
    ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1).to(self.device)) # (B, 256)
    input = x # 16, 4

    if not (1 <= self.layers <= 3):
        raise ValueError("the model pruning factor `L` should be 1 <= L <= 3")


    ## --- L = 1 ---
    x_00 = self.up_00(x, ctx_emb)  # (B, 16)
    x_10, x_01 = self.up_00_10(x_00, ctx_emb)  # (B, 64), (B, 16)
    output_01 = self.prediction(x_01)

    if self.layers == 1:
      return output_01
    
    ## --- L = 2 ---
    x_20, x_11 = self.up_10_20(x_10, ctx_emb)  # (B, 128), (B, 64)
    x_02 = self.down_02(x_11, [x_00, x_01], ctx_emb) # (B, 16)
    output_02 = self.prediction(x_02)

    if self.layers == 2 and self.deep_supervision:
      return (output_02 + output_01)/2
    elif self.layers == 2:
      return output_02
    
    
    ## --- L = 3 ---
    x_30, x_21 = self.up_20_30(x_20, ctx_emb)  # (B, 256), (B, 128)
    x_12 = self.down_12(x_21, [x_10, x_11], ctx_emb) # (B, 64)
    x_03 = self.down_03(x_12, [x_00, x_01, x_02], ctx_emb) # (B, 16)
    output_03 = self.prediction(x_03)
    
    if self.layers == 3 and self.deep_supervision:
      return (output_03 + output_02 + output_01)/3
    elif self.layers == 3:
      return output_03


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