"""
Config for Consistency Policy distillation from robomimic DDPM teachers.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class ConsistencyPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "consistency_policy"

    def algo_config(self):
        super(ConsistencyPolicyConfig, self).algo_config()

        # Consistency distillation defaults. These are intentionally conservative:
        # DDPM teacher, uniform time sampling, DDIM-style deterministic transition.
        self.algo.consistency.teacher_checkpoint_path = None
        self.algo.consistency.use_teacher_ema = True
        self.algo.consistency.warm_start = True
        self.algo.consistency.warm_start_obs_encoder = True
        self.algo.consistency.drop_teacher_after_deserialize = True

        # EMA target used by CTM loss.
        self.algo.consistency.ema_decay = 0.999
        self.algo.consistency.use_ema_inference = True

        # CTM / DSM objectives.
        self.algo.consistency.loss.ctm_weight = 1.0
        self.algo.consistency.loss.dsm_weight = 1.0
        self.algo.consistency.loss.delta = -1.0

        # Time sampling and teacher transition.
        self.algo.consistency.time_sampler = "uniform"
        self.algo.consistency.teacher_steps = 1
        self.algo.consistency.clip_sample = True

        # Inference. num_steps=1 is one-shot; 2 or 3 enables test-time chaining.
        self.algo.consistency.inference.num_steps = 1
        self.algo.consistency.inference.chaining_timesteps = []

        # Optional Karras-style loss weighting over DDPM sigma(t).
        self.algo.consistency.karras_weighting.enabled = False
        self.algo.consistency.karras_weighting.data_std = 0.5
        self.algo.consistency.karras_weighting.normalize = True
