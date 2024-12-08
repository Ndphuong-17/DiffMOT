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

        self.final_transform = MLP(in_features=out_features, out_features=out_features)
    def forward(self, mid, ctx):
        # Add final transformation
        output = self.final_transform(mid)  # Add residual from last downscaled input

        return output



class SimpleReUNet2Plus(nn.Module):
    def __init__(self, 
                 noise_dim=4, 
                 num_layers=1, 
                 hidden_size=256, 
                 filters=None, 
                 L=4,
                 deep_supervision=False):
        super(SimpleReUNet2Plus, self).__init__()
        if filters is None:
            filters = [16, 64, 128, 256, 512, 1024, 2048, 4096]
        if not (1 <= L <= len(filters)):
            raise ValueError(f"`L` must be between 1 and {len(filters)}.")
        
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.filters = filters
        self.L = L
        self.deep_supervision = deep_supervision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shared_ctx_mlp = MLP(in_features=hidden_size + 3, out_features=hidden_size)
        self.prediction = MLP(in_features=filters[0], out_features=noise_dim)

        # Define upsampling and downsampling layers
        self.up_layers = nn.ModuleList([
            TransAoA(input_size=(4 if i == 0 else filters[i - 1]), 
                     output_size=filters[i], 
                     num_layers=num_layers)
            for i in range(L+1)
        ])
        
        self.down_mid_layers = nn.ModuleList([
            MidTriangle1(in_features=filters[i+1], 
                         out_features=filters[i], 
                         num_nodes=j + 1)
            for j in range(1, L+1) for i in range(0, L-j+1)
        ])
        
        # self.down_layers = nn.ModuleList([
        #     DownTriangle1(out_features=filters[i], num_layers=num_layers)
        #     for i in range(L)
        # ])
    
        self.down_layers = nn.ModuleList([
            MLP(in_features=filters[i], out_features=filters[i])
            for i in range(L)
        ])
    
    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1)  # (B, 1)
        context = context.view(batch_size, -1)  # (B, F)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
        ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1).to(self.device))  # (B, 256)

        # --- Upsampling ---
        up_outputs = []
        current_x = x
        for i, up_layer in enumerate(self.up_layers):
            current_x = up_layer(current_x, ctx_emb)
            up_outputs.append(current_x)

        # --- Downsampling ---
        inputs = []
        for j in range(self.L):
          current_x = self.down_mid_layers[j](up_outputs[j+1], [up_outputs[j]], ctx_emb)
          current_x = self.down_layers[j](current_x)
          inputs.append([up_outputs[j], current_x])
            
            
        key = self.L
        for j in range(self.L - 1, 0, -1): # 2, 1
            for i in range(j):
              # print("i: ", i)
              # print("j: ", j)
              # print("filters[i]: ", self.filters[i])
              # print("key: ", key)
              # print(f"inputs[i]: {len(inputs[i])}, {inputs[i][0].shape}")
              # print(f"inputs[i+1]: {len(inputs[i+1])}, {inputs[i+1][-1].shape}")
              # print(self.down_mid_layers[key])
              current_x = self.down_mid_layers[key](inputs[i+1][-1], inputs[i], ctx_emb)
              # print("current_x: ", current_x.shape)
              # print(self.down_layers[i])
              current_x = self.down_layers[i](current_x)
              inputs[i][-1] = current_x
              # print(f"inputs[{i}]: {len(inputs[i])}")
              key += 1
              # print("================================================================")

        # --- Final Output ---
        # predictions = [self.prediction(down) for down in down_outputs[::-1]]
        # if self.deep_supervision:
        #     return sum(predictions) / len(predictions)
        return self.prediction(inputs[0][-1])


    



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
