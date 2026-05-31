"""
Non-invasive injection of a few-step sampler into an already-loaded
Diffusion Policy. The training code (robomimic) is never modified.

History:
  Originally, robomimic's `DiffusionPolicyUNet.deserialize()` on diffusers 0.11.1
  did not load `model["ema"]`, so we restored the saved EMA weights manually after
  `policy_from_checkpoint`. After the diffusers 0.11.1 -> 0.18.2 upgrade
  (origin/main commit c89ba4b), the new `deserialize()` DOES load EMA via
  `self.ema.load_state_dict(model["ema"])`, but the format changed: it now
  expects an EMAModel state_dict (with `shadow_params`), whereas existing
  checkpoints trained under 0.11.1 still hold the legacy format (a model-style
  state_dict whose keys match `nets.state_dict()`).

  We support both:
    - new-format checkpoints: deserialize handles them natively; restore_ema()
      is effectively a no-op.
    - legacy-format checkpoints (everything trained before the upgrade): we
      strip the legacy ema before policy_from_checkpoint (so deserialize
      doesn't trip on the wrong format), then migrate it into the new
      EMAModel.shadow_params via parameter order.

Use `load_ckpt_dict(path)` to read a checkpoint, then call
`policy_from_checkpoint(ckpt_dict=...)`, then `restore_ema(policy, ckpt_dict)`,
then `apply_sampler(...)`.
"""
import torch


LEGACY_EMA_KEY = "__legacy_ema__"


def _is_legacy_ema(ema):
    """Old format = model-style state_dict (no `shadow_params`).
    New format = EMAModel state_dict containing a `shadow_params` list.
    """
    if not isinstance(ema, dict):
        return False
    return "shadow_params" not in ema


def load_ckpt_dict(path):
    """Load a .pth checkpoint and, if it holds the legacy EMA format, move that
    blob out of model["ema"] into a stash key so robomimic's new deserialize()
    doesn't try to load it as a new-style EMAModel state. The stash is recovered
    in restore_ema(). New-format checkpoints are returned unchanged.
    """
    # weights_only=False: pytorch 2.6+ default is True, but robomimic ckpts contain
    # numpy scalars in their config metadata. Our own trusted checkpoints, safe to opt out.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ema = ckpt.get("model", {}).get("ema", None)
    if ema is not None and _is_legacy_ema(ema):
        ckpt[LEGACY_EMA_KEY] = ckpt["model"].pop("ema")
    return ckpt


def restore_ema(rollout_policy, ckpt_dict):
    """Ensure the policy's EMA holds the saved weights.

    For new-format checkpoints this is a no-op (deserialize already loaded EMA).
    For legacy-format checkpoints (stashed by load_ckpt_dict), we migrate the
    legacy weights into the new EMAModel.shadow_params.

    Returns True if a legacy EMA was injected, False otherwise.
    """
    from diffusers.training_utils import EMAModel

    dp = rollout_policy.policy  # DiffusionPolicyUNet
    legacy = ckpt_dict.get(LEGACY_EMA_KEY, None)
    if legacy is None:
        return False

    if getattr(dp, "ema", None) is None:
        # EMA disabled in this config -- load the weights directly into nets,
        # since inference will use nets as-is.
        dp.nets.load_state_dict(legacy)
        return True

    # Legacy ema is a state_dict matching nets.state_dict(). EMAModel.shadow_params
    # is indexed by parameter order (not by name), so we route through nets:
    # load legacy weights into nets, snapshot them as shadow_params via a fresh
    # EMAModel, then restore nets to its pre-EMA weights.
    nets_backup = {k: v.detach().clone() for k, v in dp.nets.state_dict().items()}
    dp.nets.load_state_dict(legacy)
    new_ema = EMAModel(parameters=dp.nets.parameters(), power=dp.algo_config.ema.power)
    dp.ema = new_ema
    dp.nets.load_state_dict(nets_backup)
    return True


def apply_sampler(rollout_policy, name, steps, base=None, clip_sample=True, solver_order=2):
    """Swap the policy's diffusion sampler in place and force a fixed step count.

    Unchanged across the diffusers 0.11.1 -> 0.18.2 upgrade: the inference loop
    in `_get_action_trajectory` still calls `noise_scheduler.set_timesteps` and
    `noise_scheduler.step` the same way, around the new EMA store/copy_to/restore
    pattern (which we don't touch).

    For UniPC at low step counts, solver_order is automatically capped at `steps`.
    """
    from .samplers import make_scheduler

    dp = rollout_policy.policy
    if base is None:
        d = dp.algo_config.ddpm
        base = dict(
            num_train_timesteps=d.num_train_timesteps,
            beta_schedule=d.beta_schedule,
            prediction_type=d.prediction_type,
        )

    eff_order = min(solver_order, steps)
    sched = make_scheduler(name, base, clip_sample=clip_sample, solver_order=eff_order)
    _orig_set = sched.set_timesteps

    def _forced_set_timesteps(num_inference_steps=None, device=None, **kw):
        if device is not None:
            return _orig_set(steps, device=device)
        return _orig_set(steps)

    sched.set_timesteps = _forced_set_timesteps
    dp.noise_scheduler = sched

    return {
        "sampler": name,
        "steps": steps,
        "base": base,
        "clip_sample": clip_sample,
        "solver_order": eff_order,
    }
