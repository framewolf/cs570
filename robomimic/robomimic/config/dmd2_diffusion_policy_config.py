"""
Config for DMD2-style Diffusion Policy distillation.
"""

from copy import deepcopy

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DMD2DiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "dmd2_diffusion_policy"

    def algo_config(self):
        super(DMD2DiffusionPolicyConfig, self).algo_config()

        self.algo.optim_params.fake_score = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.fake_score.learning_rate.initial = 1e-4

        self.algo.dmd2.teacher.ckpt_path = None
        self.algo.dmd2.teacher.use_ema = True

        self.algo.dmd2.init_student_from_teacher = True
        self.algo.dmd2.init_fake_score_from_teacher = True

        self.algo.dmd2.student.num_steps = 4
        self.algo.dmd2.student.timesteps = [99, 74, 49, 24]
        self.algo.dmd2.student.action_clip = 1.0

        self.algo.dmd2.fake_score.updates_per_generator = 5

        self.algo.dmd2.dm.loss_weight = 1.0
        self.algo.dmd2.dm.min_step = 2
        self.algo.dmd2.dm.max_step = -1
        self.algo.dmd2.dm.grad_normalizer_eps = 1e-6

        self.algo.dmd2.gan.enabled = False
        self.algo.dmd2.gan.generator_loss_weight = 0.01
        self.algo.dmd2.gan.discriminator_loss_weight = 1.0
        self.algo.dmd2.gan.min_step = 0
        self.algo.dmd2.gan.max_step = -1
