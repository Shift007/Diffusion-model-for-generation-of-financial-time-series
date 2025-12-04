import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)
        
        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        
        out = out.reshape(b, -1, h, w)
        return self.to_out(out) + x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Down stages
        self.downs = nn.ModuleList([])
        dims = [base_ch, base_ch*2, base_ch*4]
        for i in range(len(dims)-1):
            dim_in = dims[i]
            dim_out = dims[i+1]
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim),
                # Downsample only time dimension (stride=(1, 2)) to preserve frequency resolution (9 levels)
                nn.Conv2d(dim_in, dim_out, 3, stride=(1, 2), padding=1) 
            ]))

        # Middle stage
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim)

        # Up stages
        self.ups = nn.ModuleList([])
        reversed_dims = list(reversed(dims))
        for i in range(len(reversed_dims)-1):
            dim_in = reversed_dims[i]
            dim_out = reversed_dims[i+1]
            self.ups.append(nn.ModuleList([
                # Upsample only time dimension
                nn.ConvTranspose2d(dim_in, dim_out, 3, stride=(1, 2), padding=1, output_padding=(0, 1)), 
                ResnetBlock(dim_out * 2, dim_out, time_emb_dim), # *2 for skip connection
                ResnetBlock(dim_out, dim_out, time_emb_dim)
            ]))

        # Final
        self.final_res_block = ResnetBlock(base_ch, base_ch, time_emb_dim)
        self.final_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t = self.time_emb(t)
        
        x = self.init_conv(x)
        h = [x]

        # Down
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Up
        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            # Skip connection
            skip = h.pop()
            
            # Handle potential shape mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

if __name__ == '__main__':
    # Test with the specific shape from the project (N, 3, 9, 256)
    model = UNet(in_channels=3)
    x = torch.randn(2, 3, 9, 256)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Output shape mismatch!"
