"""
Sampler factory for few-step Diffusion Policy evaluation.

The trained Diffusion Policy uses diffusers schedulers through a single,
sampler-agnostic inference loop (`DiffusionPolicyUNet._get_action_trajectory`):

    self.noise_scheduler.set_timesteps(N)
    for k in self.noise_scheduler.timesteps:
        eps = noise_pred_net(sample, k, global_cond)
        sample = self.noise_scheduler.step(eps, k, sample).prev_sample

So swapping the sampler at evaluation time is just a matter of replacing
`noise_scheduler` with another object exposing the same interface. This module
builds those objects, configured to MATCH the noise schedule the policy was
trained with (otherwise the model sees noise levels it never learned).

- DDPM / DDIM come from the installed diffusers (0.11.1).
- UniPC is vendored in `unipc_scheduler.py` (diffusers 0.11.1 has no UniPC).

Training-free: the same checkpoint is reused; only the ODE/SDE solver changes.
"""
from diffusers import DDPMScheduler, DDIMScheduler

from .unipc_scheduler import UniPCMultistepScheduler


def make_scheduler(name, base, clip_sample=True, solver_order=2):
    """
    Args:
        name (str): one of {"ddpm", "ddim", "unipc"}.
        base (dict): noise-schedule kwargs taken from the trained policy's config,
            i.e. {num_train_timesteps, beta_schedule, prediction_type}. These MUST
            match training.
        clip_sample (bool): clamp the predicted sample to [-1, 1] each step (the
            policy's actions are normalized to [-1, 1]). DDPM/DDIM support this
            natively; the vendored UniPC gained an equivalent option.
        solver_order (int): multistep order for UniPC (ignored by DDPM/DDIM).
            Callers should pass min(2, steps) when probing very low step counts.

    Returns:
        a diffusers-style scheduler instance.
    """
    name = name.lower()
    if name == "ddpm":
        return DDPMScheduler(clip_sample=clip_sample, **base)
    if name == "ddim":
        return DDIMScheduler(
            clip_sample=clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            **base,
        )
    if name == "unipc":
        return UniPCMultistepScheduler(
            solver_order=solver_order,   # recommended 2 for guided/conditional sampling
            predict_x0=True,             # data-prediction form (DPM-Solver++ style), stable in few steps
            solver_type="bh2",
            clip_sample=clip_sample,
            clip_sample_range=1.0,
            **base,
        )
    raise ValueError(f"unknown sampler '{name}' (expected ddpm|ddim|unipc)")
