from diffusers import DiffusionPipeline, DDPMPipeline
import torch
import torch.nn.functional as F

from abdiff.abdiff_sampling.nn.mynet.transformer2D import Transformer2DModel


# ==========================================================
# Load pretrained diffusion pipeline
# ==========================================================
# Original:
# pipeline = DiffusionPipeline.from_pretrained("/home/pengchao/data/abdiff/train/checkpoints/20250103_1_a_1", use_safetensors=True)
# Changed to relative checkpoint path inside repo
pipeline = DiffusionPipeline.from_pretrained("checkpoints/abdiff/20250103_1_a_1", use_safetensors=True)
pipeline.scheduler.config.clip_sample = False
# generator = torch.manual_seed(0)


def sample(s, z):
    """
    Simple sampling procedure without mask (baseline diffusion sampling).
    """
    with torch.no_grad():
        device = "cuda:0"
        s = s.to(device)
        pipeline.to(device)
        z = z.to(device)

        total_timesteps = pipeline.scheduler.config.num_train_timesteps
        total_timesteps = 1000
        timesteps = torch.tensor([total_timesteps - 1], device=device)
        noise = torch.randn_like(z, device=device)
        image = pipeline.scheduler.add_noise(z, noise, timesteps)

        for t in pipeline.progress_bar(pipeline.scheduler.timesteps[-total_timesteps:]):
            t = t.to(device)
            # 1. predict noise model_output
            model_output = pipeline.unet(image, t, encoder_hidden_states=s).sample
            # 2. compute previous image: x_t -> x_t-1
            image = pipeline.scheduler.step(model_output, t, image).prev_sample
            if t % 40 == 0:
                a = 1

        z = image
        return z


def repaint_sample(s, z, cdr_mask, n_sampling_steps=1, device="cuda:0"):
    """
    Repaint sampling with CDR-H3 mask.
    Keeps framework regions fixed while resampling the loop region.
    """
    with torch.no_grad():
        s = s.to(device)
        pipeline.to(device)

        z = z.to(device)
        cdr_mask = cdr_mask.to(device)
        cdr_mask = 1 - cdr_mask
        cdr_mask = cdr_mask[..., None, :] * cdr_mask[..., None]
        cdr_mask = 1 - cdr_mask
        cdr_mask = cdr_mask.unsqueeze(2)

        total_timesteps = pipeline.scheduler.config.num_train_timesteps
        total_timesteps = 1000
        timesteps = torch.tensor([total_timesteps - 1], device=device)
        noise = torch.randn_like(z, device=device)
        image = pipeline.scheduler.add_noise(z, noise, timesteps)

        for t in pipeline.progress_bar(pipeline.scheduler.timesteps[-total_timesteps:]):
            t = t.to(device)
            for u in range(n_sampling_steps):
                # 1. predict noise model_output
                model_output = pipeline.unet(image, t, encoder_hidden_states=s).sample
                # 2. compute previous image: x_t -> x_t-1
                image_t = pipeline.scheduler.step(model_output, t, image).prev_sample
                noise_t = torch.randn_like(z, device=device)
                noise_image_t = pipeline.scheduler.add_noise(z, noise_t, t)
                image = image_t * cdr_mask + noise_image_t * (1 - cdr_mask)
                if t > 0 and u < n_sampling_steps - 1:
                    image = _get_x_t_from_x_t_minus1(image, noise_t, t)
                if t % 40 == 0:
                    a = 1

        z = image
        return z


def _get_x_t_from_x_t_minus1(x_t_minus1, noise_t, t):
    """
    Internal helper to recover x_t from x_t-1 during diffusion.
    """
    noise_t = torch.randn_like(x_t_minus1)
    alpha_t = pipeline.scheduler.alphas[t]
    x_t = torch.sqrt(alpha_t) * x_t_minus1 + torch.sqrt(1 - alpha_t) * noise_t
    return x_t
