import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal time embedding as in DDPM
import math
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device).float() * -(math.log(10000.0) / (half_dim - 1))
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def conv_block(in_ch, out_ch, time_emb_dim=None):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    ]
    if time_emb_dim is not None:
        layers.append(nn.Conv2d(time_emb_dim, out_ch, kernel_size=1))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        if t_emb is not None:
            # t_emb: (B, time_emb_dim)
            t = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.res_conv(x))



class UNet(nn.Module):
    """A small U-Net for diffusion denoising with explicit time embedding injection."""
    def __init__(self, in_channels=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Down path: use only ResidualBlock, not Sequential
        self.inc = ResidualBlock(in_channels, base_ch, time_emb_dim)
        self.down1 = ResidualBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down2 = ResidualBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # Up path: use only ResidualBlock, not Sequential
        # Channel counts: up2 gets cat([upsampled xm (base_ch*4), x2 (base_ch*2)]) = base_ch*6
        # up1 gets cat([upsampled u2 (base_ch*2), x1 (base_ch)]) = base_ch*3
        self.up2 = ResidualBlock(base_ch * 6, base_ch * 2, time_emb_dim)
        self.up1 = ResidualBlock(base_ch * 3, base_ch, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Output convolution: only takes x, so Sequential is fine
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 1)
        )

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B,) integer timesteps
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        # Down path with time embedding injection
        x1 = self.inc(x, t_emb)
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        xm = self.mid(x3, t_emb)

        # Up path with skip connections and time embedding injection
        u2 = self.up2(torch.cat([F.interpolate(xm, size=x2.shape[2:], mode='nearest'), x2], dim=1), t_emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=x1.shape[2:], mode='nearest'), x1], dim=1), t_emb)

        # Only pass u1 to out_conv (no t_emb)
        out = self.out_conv(u1)
        return out


if __name__ == '__main__':
    # quick shape test
    model = UNet(in_channels=1)
    x = torch.randn(2, 1, 6, 64)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    print(y.shape)
