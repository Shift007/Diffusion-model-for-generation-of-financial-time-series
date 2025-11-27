import math
import torch
import torch.nn.functional as F



def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better for preventing mode collapse
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Diffusion:
    def __init__(self, model, timesteps=1000, device='cpu', beta_schedule='cosine'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        if beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, timesteps, device=device)
        elif beta_schedule == 'quadratic':  # NUOVO - più aggressivo per le tails
            self.betas = torch.linspace(0.0001**0.5, 0.02**0.5, timesteps, device=device) ** 2
        elif beta_schedule == 'sigmoid':  # NUOVO - ancora più aggressivo
            betas = torch.linspace(-6, 6, timesteps)
            self.betas = torch.sigmoid(betas) * (0.02 - 0.0001) + 0.0001
            self.betas = self.betas.to(device)
        
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]], dim=0)

        # Precompute useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # Add noise to x_start at timestep t
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        if loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(predicted_noise, noise, reduction='none')
        elif loss_type == 'weighted_mse':  # NUOVO
            # Peso maggiore per valori estremi
            weights = 1.0 + torch.abs(x_start)  # Peso proporzionale al valore
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
            loss = loss * weights
        elif loss_type == 'tail_focused':  # NUOVO - focus sulle tails
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
            # Peso esponenziale per valori estremi
            threshold = 1.5  # Soglia per considerare "tail"
            tail_mask = (torch.abs(x_start) > threshold).float()
            tail_weight = 3.0  # Peso 3x per le tails
            weights = 1.0 + tail_mask * (tail_weight - 1.0)
            loss = loss * weights
        
        return loss.mean()
    @torch.no_grad()
    def p_sample(self, x, t_index, temperature=1.0):
        # Single reverse step for batch x at timestep t_index (int index)
        betas_t = self.betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        alpha_t = self.alphas[t_index]

        # Predict noise
        t = torch.full((x.shape[0],), t_index, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x, t)

        # Compute posterior mean
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (betas_t / sqrt_one_minus_alpha_cumprod_t)
        mean = coef1 * (x - coef2 * predicted_noise)

        if t_index == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            var = self.posterior_variance[t_index]
            # Apply temperature scaling to variance
            return mean + torch.sqrt(var * temperature) * noise

    @torch.no_grad()
    def sample(self, shape, temperature=1.5):
        """Sample with temperature control for diversity"""
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            x = self.p_sample(x, i, temperature=temperature)
        return x


def sample_loop(model, image_size, channels=1, timesteps=1000, device='cpu', batch_size=8):
    diff = Diffusion(model, timesteps=timesteps, device=device)
    samples = diff.sample((batch_size, channels, image_size, image_size))
    return samples
