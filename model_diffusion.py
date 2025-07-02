import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

# --- ResNet Block ---
class ResBlock(nn.Module):
    def __init__(self, channels, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.act1 = nn.SiLU()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.norm2 = nn.GroupNorm(32, channels)
        self.act2 = nn.SiLU()
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.time_emb_proj = nn.Linear(time_emb_dim, channels) if time_emb_dim is not None else None

    def forward(self, x, t_emb=None):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        if self.time_emb_proj is not None and t_emb is not None:
            h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        return (x + h) / math.sqrt(2)

# --- Self-Attention Block ---
class SelfAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.k = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.v = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.proj = spectral_norm(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.k(h).reshape(B, C, H * W)                   # (B, C, HW)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)     # (B, HW, HW)
        out = attn @ v                                       # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return (x + self.proj(out)) / math.sqrt(2)

# --- Sinusoidal Time Embedding ---
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1,0,0))
    return emb

# --- Down/Up Blocks for U-Net ---
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks, time_emb_dim, attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.resblocks = nn.ModuleList([ResBlock(out_ch, time_emb_dim) for _ in range(num_res_blocks)])
        self.attn = SelfAttention2d(out_ch) if attn else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv_in(x)
        for block in self.resblocks:
            h = block(h, t_emb)
        h = self.attn(h)
        return h

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks, time_emb_dim, attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.resblocks = nn.ModuleList([ResBlock(out_ch, time_emb_dim) for _ in range(num_res_blocks)])
        self.attn = SelfAttention2d(out_ch) if attn else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv_in(x)
        for block in self.resblocks:
            h = block(h, t_emb)
        h = self.attn(h)
        return h

# --- SR3-like U-Net ---
class SR3UNet(nn.Module):
    def __init__(
        self,
        in_channels=6,  # noisy HR + upsampled LR
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 4, 8, 8),
        num_res_blocks=3,
        attn_resolutions=(16,),  # Only at 16x16
        time_emb_dim=512
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Downsampling
        self.downs = nn.ModuleList()
        channels = [base_channels * m for m in channel_mults]
        in_ch = in_channels
        for i, ch in enumerate(channels):
            attn = 2**(len(channel_mults)-i-1) in attn_resolutions
            self.downs.append(DownBlock(in_ch, ch, num_res_blocks, time_emb_dim, attn=attn))
            in_ch = ch

        # Middle
        self.mid = nn.Sequential(
            ResBlock(channels[-1], time_emb_dim),
            SelfAttention2d(channels[-1]),
            ResBlock(channels[-1], time_emb_dim)
        )

        # Upsampling
        self.ups = nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            attn = 2**(len(channel_mults)-i-1) in attn_resolutions
            self.ups.append(UpBlock(ch*2 if i != len(channels)-1 else ch, ch, num_res_blocks, time_emb_dim, attn=attn))

        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        # x: (B, 6, H, W)  (noisy HR + upsampled LR)
        # t: (B,) or (B, 1)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Down path
        hs = []
        h = x
        for down in self.downs:
            h = down(h, t_emb)
            hs.append(h)
            h = F.avg_pool2d(h, 2)

        # Middle
        h = self.mid(h, t_emb)

        # Up path
        for up in self.ups:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, hs.pop()], dim=1) / math.sqrt(2)
            h = up(h, t_emb)

        return self.final(h)

class DiffusionModel(nn.Module):
    def __init__(self, time_steps, 
                 beta_start = 10e-4, 
                 beta_end = 0.02,
                 image_dims = (3, 128, 128)):
        super().__init__()
        self.time_steps = time_steps
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)
        # For SR3, input is (noisy HR + upsampled LR)
        self.model = SR3UNet(
            in_channels=2*c,
            out_channels=c,
            base_channels=128,
            channel_mults=(1, 2, 4, 8, 8),  # for 128x128 input
            num_res_blocks=3,
            attn_resolutions=(16,),  # Only at 16x16
            time_emb_dim=512
        )

    def add_noise(self, x, ts):
        noise = torch.randn_like(x)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x, t):
        return self.model(x, t)


