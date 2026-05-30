"""
DMD2-style distillation for low-dimensional Diffusion Policy.

This adapts the DMD2 training recipe to action trajectories. The student is a
few-step diffusion sampler over actions, the frozen teacher estimates the real
score, and a trainable fake-score diffusion model tracks the student's output
distribution.
"""

import os
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import replace_bn_with_gn


@register_algo_factory_func("dmd2_diffusion_policy")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return DMD2DiffusionPolicyUNet, {}
    raise RuntimeError("DMD2 diffusion policy currently requires the UNet policy.")


class DMD2DiffusionPolicyUNet(PolicyAlgo):
    """
    Distills a many-step Diffusion Policy teacher into a fixed few-step student.

    The student and fake-score models use the same ConditionalUnet1D backbone as
    the original Diffusion Policy. The student is sampled with a deterministic
    DDIM-style update over a fixed timestep schedule.
    """

    def _create_policy_nets(self, include_gan_head=False):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
            self.obs_config.encoder
        )

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)
        obs_dim = obs_encoder.output_shape()[0]

        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=list(self.algo_config.unet.down_dims),
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        modules = {
            "obs_encoder": obs_encoder,
            "noise_pred_net": noise_pred_net,
        }
        if include_gan_head:
            gan_dim = list(self.algo_config.unet.down_dims)[-1]
            modules["gan_head"] = nn.Sequential(
                nn.Linear(gan_dim, gan_dim),
                nn.SiLU(),
                nn.Linear(gan_dim, 1),
            )

        return nn.ModuleDict(modules)

    def _scheduler_cfg(self):
        for name in ("ddpm", "ddim", "deis", "dpm_solver"):
            if name in self.algo_config and self.algo_config[name].enabled:
                cfg = self.algo_config[name]
                return cfg.num_train_timesteps, cfg.beta_schedule, cfg.prediction_type
        raise RuntimeError("No diffusion scheduler is enabled in the config.")

    def _create_networks(self):
        nets = nn.ModuleDict(
            {
                "policy": self._create_policy_nets(),
                "fake_score": self._create_policy_nets(include_gan_head=True),
            }
        )
        self.nets = nets.float().to(self.device)

        num_train_timesteps, beta_schedule, prediction_type = self._scheduler_cfg()
        if prediction_type != "epsilon":
            raise ValueError("DMD2 action distillation currently expects epsilon prediction.")

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type=prediction_type,
        )

        self.ema = None
        if self.algo_config.ema.enabled:
            self.ema = EMAModel(
                parameters=self.nets["policy"].parameters(),
                power=self.algo_config.ema.power,
            )

        self.teacher_nets = None
        self._train_step = 0
        self._last_policy_update = False
        self._last_fake_update = False
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.student_timesteps = self._make_student_timesteps()

        teacher_state = self._load_teacher_state_for_initialization()
        if teacher_state is not None:
            if self.algo_config.dmd2.init_student_from_teacher:
                self.nets["policy"].load_state_dict(teacher_state, strict=True)
                self._reset_policy_ema()
            if self.algo_config.dmd2.init_fake_score_from_teacher:
                self.nets["fake_score"].load_state_dict(teacher_state, strict=False)

    def _reset_policy_ema(self):
        if self.ema is not None:
            self.ema = EMAModel(
                parameters=self.nets["policy"].parameters(),
                power=self.algo_config.ema.power,
            )

    def _make_student_timesteps(self):
        raw_timesteps = list(self.algo_config.dmd2.student.timesteps)
        if len(raw_timesteps) == 0:
            num_steps = int(self.algo_config.dmd2.student.num_steps)
            max_t = int(self.noise_scheduler.config.num_train_timesteps) - 1
            raw_timesteps = torch.linspace(max_t, 0, num_steps + 1)[:-1].long().tolist()

        timesteps = [int(t) for t in raw_timesteps]
        if any(timesteps[i] <= timesteps[i + 1] for i in range(len(timesteps) - 1)):
            raise ValueError("DMD2 student timesteps must be strictly descending.")
        max_t = int(self.noise_scheduler.config.num_train_timesteps) - 1
        if min(timesteps) < 0 or max(timesteps) > max_t:
            raise ValueError("DMD2 student timesteps must be in the scheduler range.")
        return timesteps

    def _torch_load_checkpoint(self, ckpt_path):
        ckpt_path = os.path.expandvars(os.path.expanduser(ckpt_path))
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")

    def _teacher_ckpt_path(self):
        ckpt_path = self.algo_config.dmd2.teacher.ckpt_path
        if ckpt_path is None or ckpt_path == "":
            return None
        return os.path.expandvars(os.path.expanduser(ckpt_path))

    def _load_teacher_state_for_initialization(self):
        ckpt_path = self._teacher_ckpt_path()
        if ckpt_path is None:
            return None
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("DMD2 teacher checkpoint not found: {}".format(ckpt_path))

        ckpt = self._torch_load_checkpoint(ckpt_path)
        policy_state = {
            k[len("policy."):]: v
            for k, v in ckpt["model"]["nets"].items()
            if k.startswith("policy.")
        }

        if self.algo_config.dmd2.teacher.use_ema and ckpt["model"].get("ema", None) is not None:
            teacher_nets = nn.ModuleDict({"policy": self._create_policy_nets()})
            teacher_nets = teacher_nets.float().to(self.device)
            teacher_nets.load_state_dict(ckpt["model"]["nets"], strict=True)
            teacher_ema = EMAModel(
                parameters=teacher_nets.parameters(),
                power=ckpt["model"]["ema"].get("power", self.algo_config.ema.power),
            )
            teacher_ema.load_state_dict(ckpt["model"]["ema"])
            teacher_ema.copy_to(teacher_nets.parameters())
            policy_state = teacher_nets["policy"].state_dict()

        return policy_state

    def _ensure_teacher_loaded(self):
        if self.teacher_nets is not None:
            return

        ckpt_path = self._teacher_ckpt_path()
        if ckpt_path is None:
            raise RuntimeError("DMD2 teacher checkpoint path is required for training.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("DMD2 teacher checkpoint not found: {}".format(ckpt_path))

        ckpt = self._torch_load_checkpoint(ckpt_path)
        self.teacher_nets = nn.ModuleDict({"policy": self._create_policy_nets()})
        self.teacher_nets = self.teacher_nets.float().to(self.device)
        self.teacher_nets.load_state_dict(ckpt["model"]["nets"], strict=True)

        if self.algo_config.dmd2.teacher.use_ema and ckpt["model"].get("ema", None) is not None:
            teacher_ema = EMAModel(
                parameters=self.teacher_nets.parameters(),
                power=ckpt["model"]["ema"].get("power", self.algo_config.ema.power),
            )
            teacher_ema.load_state_dict(ckpt["model"]["ema"])
            teacher_ema.copy_to(self.teacher_nets.parameters())

        self.teacher_nets.eval()
        self.teacher_nets.requires_grad_(False)

    def _encode_obs(self, batch_or_obs, nets, goal_obs=None):
        inputs = {"obs": batch_or_obs["obs"] if "obs" in batch_or_obs else batch_or_obs, "goal": goal_obs}
        if "goal_obs" in batch_or_obs:
            inputs["goal"] = batch_or_obs["goal_obs"]

        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert obs_features.ndim == 3
        return obs_features.flatten(start_dim=1)

    def _extract_alpha(self, timesteps, target):
        alphas = self.noise_scheduler.alphas_cumprod.to(
            device=target.device,
            dtype=target.dtype,
        )
        out = alphas[timesteps]
        while out.ndim < target.ndim:
            out = out.unsqueeze(-1)
        return out

    def _predict_x0_from_epsilon(self, noisy_actions, noise_pred, timesteps):
        alpha_prod_t = self._extract_alpha(timesteps, noisy_actions)
        beta_prod_t = 1.0 - alpha_prod_t
        return (noisy_actions - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

    def _ddim_prev_sample(self, sample, noise_pred, timesteps, prev_timestep):
        pred_x0 = self._predict_x0_from_epsilon(sample, noise_pred, timesteps)

        if prev_timestep < 0:
            alpha_prod_prev = torch.ones_like(self._extract_alpha(timesteps, sample))
        else:
            prev = torch.full_like(timesteps, int(prev_timestep))
            alpha_prod_prev = self._extract_alpha(prev, sample)

        prev_sample = alpha_prod_prev.sqrt() * pred_x0 + (1.0 - alpha_prod_prev).sqrt() * noise_pred
        return prev_sample, pred_x0

    def _student_sample(self, obs_cond, requires_grad):
        B = obs_cond.shape[0]
        Tp = self.algo_config.horizon.prediction_horizon
        sample = torch.randn((B, Tp, self.ac_dim), device=self.device)

        context = TorchUtils.maybe_no_grad(no_grad=not requires_grad)
        with context:
            for i, timestep in enumerate(self.student_timesteps):
                t = torch.full((B,), int(timestep), device=self.device, dtype=torch.long)
                noise_pred = self.nets["policy"]["noise_pred_net"](
                    sample=sample,
                    timestep=t,
                    global_cond=obs_cond,
                )
                prev_timestep = self.student_timesteps[i + 1] if i + 1 < len(self.student_timesteps) else -1
                sample, _ = self._ddim_prev_sample(sample, noise_pred, t, prev_timestep)
                action_clip = self.algo_config.dmd2.student.action_clip
                if action_clip is not None:
                    sample = sample.clamp(-float(action_clip), float(action_clip))
        return sample

    def _sample_dmd_timesteps(self, batch_size):
        min_step = int(self.algo_config.dmd2.dm.min_step)
        max_step = int(self.algo_config.dmd2.dm.max_step)
        scheduler_max = int(self.noise_scheduler.config.num_train_timesteps) - 1
        max_step = scheduler_max if max_step < 0 else min(max_step, scheduler_max)
        return torch.randint(
            min_step,
            max_step + 1,
            (batch_size,),
            device=self.device,
        ).long()

    def _fake_score_loss(self, generated_actions, batch):
        B = generated_actions.shape[0]
        clean_actions = generated_actions.detach()
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

        obs_cond = self._encode_obs(batch, self.nets["fake_score"])
        noise_pred = self.nets["fake_score"]["noise_pred_net"](
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=obs_cond,
        )
        return F.mse_loss(noise_pred, noise)

    def _gan_enabled(self):
        return bool(self.algo_config.dmd2.gan.enabled)

    def _set_requires_grad(self, module, requires_grad):
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    def _sample_gan_timesteps(self, batch_size):
        min_step = int(self.algo_config.dmd2.gan.min_step)
        max_step = int(self.algo_config.dmd2.gan.max_step)
        scheduler_max = int(self.noise_scheduler.config.num_train_timesteps) - 1
        max_step = scheduler_max if max_step < 0 else min(max_step, scheduler_max)
        return torch.randint(
            min_step,
            max_step + 1,
            (batch_size,),
            device=self.device,
        ).long()

    def _gan_logits(self, actions, batch):
        B = actions.shape[0]
        timesteps = self._sample_gan_timesteps(B)
        if int(self.algo_config.dmd2.gan.max_step) == 0:
            noisy_actions = actions
        else:
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        obs_cond = self._encode_obs(batch, self.nets["fake_score"])
        _, bottleneck = self.nets["fake_score"]["noise_pred_net"].forward_with_bottleneck(
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=obs_cond,
        )
        pooled = bottleneck.mean(dim=-1)
        return self.nets["fake_score"]["gan_head"](pooled).squeeze(-1)

    def _gan_discriminator_loss(self, generated_actions, batch):
        real_logits = self._gan_logits(batch["actions"], batch)
        fake_logits = self._gan_logits(generated_actions.detach(), batch)

        loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        log = {
            "gan_disc_loss": loss,
            "gan_real_prob": torch.sigmoid(real_logits.detach()).mean(),
            "gan_fake_prob": torch.sigmoid(fake_logits.detach()).mean(),
        }
        return loss, log

    def _gan_generator_loss(self, generated_actions, batch):
        logits = self._gan_logits(generated_actions, batch)
        loss = F.softplus(-logits).mean()
        log = {
            "gan_gen_loss": loss,
            "gan_gen_fake_prob": torch.sigmoid(logits.detach()).mean(),
        }
        return loss, log

    def _distribution_matching_loss(self, generated_actions, batch):
        self._ensure_teacher_loaded()

        B = generated_actions.shape[0]
        with torch.no_grad():
            clean_actions = generated_actions.detach()
            noise = torch.randn_like(clean_actions)
            timesteps = self._sample_dmd_timesteps(B)
            noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

            teacher_obs_cond = self._encode_obs(batch, self.teacher_nets["policy"])
            teacher_noise = self.teacher_nets["policy"]["noise_pred_net"](
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=teacher_obs_cond,
            )
            pred_real_actions = self._predict_x0_from_epsilon(
                noisy_actions,
                teacher_noise,
                timesteps,
            )

            fake_obs_cond = self._encode_obs(batch, self.nets["fake_score"])
            fake_noise = self.nets["fake_score"]["noise_pred_net"](
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=fake_obs_cond,
            )
            pred_fake_actions = self._predict_x0_from_epsilon(
                noisy_actions,
                fake_noise,
                timesteps,
            )

            p_real = clean_actions - pred_real_actions
            p_fake = clean_actions - pred_fake_actions
            reduce_dims = tuple(range(1, p_real.ndim))
            normalizer = torch.abs(p_real).mean(dim=reduce_dims, keepdim=True)
            normalizer = normalizer.clamp_min(self.algo_config.dmd2.dm.grad_normalizer_eps)
            grad = (p_real - p_fake) / normalizer
            grad = torch.nan_to_num(grad)

        target = (generated_actions - grad).detach()
        loss = 0.5 * F.mse_loss(generated_actions.float(), target.float(), reduction="mean")
        log = {
            "dm_grad_norm": torch.norm(grad.detach()).float(),
            "dm_pred_real_l1": torch.abs(clean_actions - pred_real_actions).mean().float(),
            "dm_pred_fake_l1": torch.abs(clean_actions - pred_fake_actions).mean().float(),
        }
        return loss, log

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            if not torch.all(in_range).item():
                raise ValueError(
                    "'actions' must be in range [-1,1] for DMD2 Diffusion Policy."
                )
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DMD2DiffusionPolicyUNet, self).train_on_batch(
                batch,
                epoch,
                validate=validate,
            )

            self._last_policy_update = False
            self._last_fake_update = False
            if not validate:
                self._train_step += 1

            update_ratio = int(self.algo_config.dmd2.fake_score.updates_per_generator)
            compute_generator = validate or (self._train_step % update_ratio == 0)

            policy_obs_cond = self._encode_obs(batch, self.nets["policy"])
            generated_actions = self._student_sample(
                policy_obs_cond,
                requires_grad=(not validate and compute_generator),
            )

            fake_score_loss = self._fake_score_loss(generated_actions, batch)
            gan_disc_loss = zero = generated_actions.new_tensor(0.0)
            gan_gen_loss = zero
            gan_log = {
                "gan_disc_loss": zero,
                "gan_gen_loss": zero,
                "gan_real_prob": zero,
                "gan_fake_prob": zero,
                "gan_gen_fake_prob": zero,
            }
            if self._gan_enabled():
                gan_disc_loss, gan_disc_log = self._gan_discriminator_loss(
                    generated_actions,
                    batch,
                )
                gan_log.update(gan_disc_log)

            fake_score_grad_norms = None
            if not validate:
                critic_loss = (
                    fake_score_loss
                    + gan_disc_loss * self.algo_config.dmd2.gan.discriminator_loss_weight
                )
                fake_score_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["fake_score"],
                    optim=self.optimizers["fake_score"],
                    loss=critic_loss,
                    max_grad_norm=self.global_config.train.max_grad_norm,
                )
                self._last_fake_update = True

            dm_loss = zero
            dm_log = {
                "dm_grad_norm": zero,
                "dm_pred_real_l1": zero,
                "dm_pred_fake_l1": zero,
            }
            policy_grad_norms = None
            generator_loss = dm_loss * self.algo_config.dmd2.dm.loss_weight
            if compute_generator:
                dm_loss, dm_log = self._distribution_matching_loss(generated_actions, batch)
                generator_loss = dm_loss * self.algo_config.dmd2.dm.loss_weight
                if self._gan_enabled():
                    self._set_requires_grad(self.nets["fake_score"], False)
                    gan_gen_loss, gan_gen_log = self._gan_generator_loss(
                        generated_actions,
                        batch,
                    )
                    self._set_requires_grad(self.nets["fake_score"], True)
                    gan_log.update(gan_gen_log)
                    generator_loss = (
                        generator_loss
                        + gan_gen_loss * self.algo_config.dmd2.gan.generator_loss_weight
                    )
                if not validate:
                    policy_grad_norms = TorchUtils.backprop_for_loss(
                        net=self.nets["policy"],
                        optim=self.optimizers["policy"],
                        loss=generator_loss,
                        max_grad_norm=self.global_config.train.max_grad_norm,
                    )
                    if self.ema is not None:
                        self.ema.step(self.nets["policy"].parameters())
                    self._last_policy_update = True

            losses = {
                "loss": (
                    generator_loss
                    + fake_score_loss
                    + gan_disc_loss * self.algo_config.dmd2.gan.discriminator_loss_weight
                ),
                "dm_loss": dm_loss,
                "fake_score_loss": fake_score_loss,
                "gan_disc_loss": gan_disc_loss,
                "gan_gen_loss": gan_gen_loss,
                "student_action_abs_mean": generated_actions.detach().abs().mean(),
            }
            losses.update(dm_log)
            losses.update(gan_log)
            info["losses"] = TensorUtils.detach(losses)
            info["dmd2_update"] = {
                "policy": self._last_policy_update,
                "fake_score": self._last_fake_update,
            }
            if policy_grad_norms is not None:
                info["policy_grad_norms"] = policy_grad_norms
            if fake_score_grad_norms is not None:
                info["fake_score_grad_norms"] = fake_score_grad_norms

        return info

    def on_gradient_step(self):
        if not hasattr(self, "step_lr_schedulers_every_batch"):
            return
        if (
            self._last_policy_update
            and self.step_lr_schedulers_every_batch.get("policy", False)
            and self.lr_schedulers["policy"] is not None
        ):
            self.lr_schedulers["policy"].step()
        if (
            self._last_fake_update
            and self.step_lr_schedulers_every_batch.get("fake_score", False)
            and self.lr_schedulers["fake_score"] is not None
        ):
            self.lr_schedulers["fake_score"].step()

    def log_info(self, info):
        log = super(DMD2DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["loss"].item()
        log["DMD2/DM_Loss"] = info["losses"]["dm_loss"].item()
        log["DMD2/Fake_Score_Loss"] = info["losses"]["fake_score_loss"].item()
        log["DMD2/GAN_Disc_Loss"] = info["losses"]["gan_disc_loss"].item()
        log["DMD2/GAN_Gen_Loss"] = info["losses"]["gan_gen_loss"].item()
        log["DMD2/GAN_Real_Prob"] = info["losses"]["gan_real_prob"].item()
        log["DMD2/GAN_Fake_Prob"] = info["losses"]["gan_fake_prob"].item()
        log["DMD2/GAN_Gen_Fake_Prob"] = info["losses"]["gan_gen_fake_prob"].item()
        log["DMD2/DM_Grad_Norm"] = info["losses"]["dm_grad_norm"].item()
        log["DMD2/DM_Pred_Real_L1"] = info["losses"]["dm_pred_real_l1"].item()
        log["DMD2/DM_Pred_Fake_L1"] = info["losses"]["dm_pred_fake_l1"].item()
        log["DMD2/Student_Action_Abs_Mean"] = info["losses"]["student_action_abs_mean"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "fake_score_grad_norms" in info:
            log["Fake_Score_Grad_Norms"] = info["fake_score_grad_norms"]
        return log

    def reset(self):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        if len(self.action_queue) == 0:
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)
            self.action_queue.extend(action_sequence[0])
        action = self.action_queue.popleft()
        return action.unsqueeze(0)

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        nets = self.nets["policy"]
        if self.ema is not None:
            self.ema.store(self.nets["policy"].parameters())
            self.ema.copy_to(self.nets["policy"].parameters())

        obs_cond = self._encode_obs(obs_dict, nets, goal_obs=goal_dict)
        action_sequence = self._student_sample(obs_cond, requires_grad=False)

        if self.ema is not None:
            self.ema.restore(self.nets["policy"].parameters())

        start = To - 1
        end = start + Ta
        return action_sequence[:, start:end]

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "optimizers": TorchUtils.get_state_dict(self.optimizers),
            "lr_schedulers": TorchUtils.get_state_dict(self.lr_schedulers),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "dmd2_train_step": self._train_step,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None and self.ema is not None:
            self.ema.load_state_dict(model_dict["ema"])
        self._train_step = model_dict.get("dmd2_train_step", 0)

        if load_optimizers:
            TorchUtils.load_state_dict(self.optimizers, model_dict["optimizers"])
            TorchUtils.load_state_dict(self.lr_schedulers, model_dict["lr_schedulers"])
