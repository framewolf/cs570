"""
Consistency distillation for robomimic Diffusion Policy checkpoints.

This implementation keeps the original diffusion_policy algo untouched. It uses
a frozen robomimic DDPM teacher checkpoint, warm-starts a CTM-style student UNet
from the teacher UNet, and saves checkpoints through robomimic's normal policy
serialization path so run_trained_agent.py can evaluate them directly.
"""

import copy
import math
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.models.consistency_policy_nets as CPNets
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.algo.diffusion_policy import replace_bn_with_gn


@register_algo_factory_func("consistency_policy")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return ConsistencyPolicyUNet, {}
    raise RuntimeError("ConsistencyPolicyUNet requires algo.unet.enabled=true")


class ConsistencyPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        obs_encoder = self._make_obs_encoder()
        obs_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_dim * self.algo_config.horizon.observation_horizon

        ctm_net = CPNets.CTMConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "obs_encoder": obs_encoder,
                        "ctm_net": ctm_net,
                    }
                )
            }
        )
        self.nets = nets.float().to(self.device)
        self.target_nets = copy.deepcopy(self.nets).float().to(self.device)
        self.target_nets.requires_grad_(False)
        self.target_nets.eval()

        self.noise_scheduler = self._make_ddpm_scheduler()
        self.teacher_nets = None
        self._load_teacher()

        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def _make_obs_encoder(self):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
            self.obs_config.encoder
        )
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        return replace_bn_with_gn(obs_encoder)

    def _make_teacher_nets(self):
        obs_encoder = self._make_obs_encoder()
        obs_dim = obs_encoder.output_shape()[0]
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )
        teacher_nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "obs_encoder": obs_encoder,
                        "noise_pred_net": noise_pred_net,
                    }
                )
            }
        )
        return teacher_nets.float().to(self.device)

    def _make_ddpm_scheduler(self):
        if not self.algo_config.ddpm.enabled:
            raise RuntimeError("consistency_policy currently distills DDPM teachers")
        return DDPMScheduler(
            num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
            beta_schedule=self.algo_config.ddpm.beta_schedule,
            clip_sample=self.algo_config.ddpm.clip_sample,
            prediction_type=self.algo_config.ddpm.prediction_type,
        )

    def _load_teacher(self):
        teacher_path = self.algo_config.consistency.teacher_checkpoint_path
        if teacher_path in (None, "", "None"):
            print("No teacher checkpoint configured for consistency_policy.")
            return

        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=teacher_path)
        teacher_nets = self._make_teacher_nets()
        teacher_nets.load_state_dict(ckpt_dict["model"]["nets"])

        if (
            self.algo_config.consistency.use_teacher_ema
            and ckpt_dict["model"].get("ema", None) is not None
        ):
            self._copy_ema_state_to_module(
                ema_state=ckpt_dict["model"]["ema"],
                module=teacher_nets,
            )

        teacher_nets.eval()
        teacher_nets.requires_grad_(False)
        self.teacher_nets = teacher_nets
        print("Loaded frozen DDPM teacher from {}".format(teacher_path))

        if self.algo_config.consistency.warm_start:
            self._warm_start_from_teacher()

    @staticmethod
    def _copy_ema_state_to_module(ema_state, module):
        shadow_params = ema_state.get("shadow_params", None)
        if shadow_params is None:
            raise RuntimeError(
                "Unsupported teacher EMA state format: missing shadow_params."
            )
        params = list(module.parameters())
        if len(shadow_params) != len(params):
            raise RuntimeError(
                "Teacher EMA parameter count mismatch: {} shadow params vs {} params.".format(
                    len(shadow_params), len(params)
                )
            )
        for param, shadow in zip(params, shadow_params):
            param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))

    def _warm_start_from_teacher(self):
        if self.teacher_nets is None:
            raise RuntimeError("Cannot warm-start without a loaded teacher.")

        if self.algo_config.consistency.warm_start_obs_encoder:
            self.nets["policy"]["obs_encoder"].load_state_dict(
                self.teacher_nets["policy"]["obs_encoder"].state_dict()
            )

        teacher_sd = self.teacher_nets["policy"]["noise_pred_net"].state_dict()
        student_sd = self.nets["policy"]["ctm_net"].state_dict()
        dsed = self.algo_config.unet.diffusion_step_embed_dim

        copied = 0
        expanded = 0
        for key, value in teacher_sd.items():
            if key not in student_sd:
                continue
            if student_sd[key].shape == value.shape:
                student_sd[key] = value.detach().clone()
                copied += 1
            elif key.endswith("cond_encoder.1.weight"):
                old_cols = value.shape[1]
                new_cols = student_sd[key].shape[1]
                if new_cols == old_cols + dsed:
                    zeros = torch.zeros(
                        value.shape[0],
                        dsed,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    student_sd[key] = torch.cat(
                        [value[:, :dsed], zeros, value[:, dsed:]], dim=1
                    )
                    expanded += 1

        self.nets["policy"]["ctm_net"].load_state_dict(student_sd)
        self.target_nets.load_state_dict(self.nets.state_dict())
        print(
            "Warm-started CTM student from teacher UNet "
            "({} copied tensors, {} stop-time expansions).".format(copied, expanded)
        )

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
                    "'actions' must be in range [-1,1] for Consistency Policy."
                )
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ConsistencyPolicyUNet, self).train_on_batch(
                batch, epoch, validate=validate
            )

            losses = OrderedDict()
            total_loss = batch["actions"].new_tensor(0.0)

            ctm_weight = self.algo_config.consistency.loss.ctm_weight
            if ctm_weight > 0:
                ctm_loss = self._ctm_loss(batch)
                losses["ctm_loss"] = ctm_loss
                total_loss = total_loss + ctm_weight * ctm_loss

            dsm_weight = self.algo_config.consistency.loss.dsm_weight
            if dsm_weight > 0:
                dsm_loss = self._dsm_loss(batch)
                losses["dsm_loss"] = dsm_loss
                total_loss = total_loss + dsm_weight * dsm_loss

            losses["loss"] = total_loss
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                )
                self._update_target_nets()
                info["policy_grad_norms"] = policy_grad_norms

        return info

    def _ctm_loss(self, batch):
        if self.teacher_nets is None:
            raise RuntimeError("CTM loss requires algo.consistency.teacher_checkpoint_path")

        actions = batch["actions"]
        B = actions.shape[0]
        obs_cond = self._get_obs_cond(batch["obs"], batch["goal_obs"], self.nets)
        with torch.no_grad():
            teacher_obs_cond = self._get_obs_cond(
                batch["obs"], batch["goal_obs"], self.teacher_nets
            )
            target_obs_cond = self._get_obs_cond(
                batch["obs"], batch["goal_obs"], self.target_nets
            )

        t, s, u = self._sample_ctm_timesteps(B, actions.device)
        noise = torch.randn_like(actions)
        x_t = self.noise_scheduler.add_noise(actions, noise, t)

        with torch.no_grad():
            eps_teacher = self.teacher_nets["policy"]["noise_pred_net"](
                x_t, t, global_cond=teacher_obs_cond
            )
            x_u = self._ddim_jump(x_t, eps_teacher, t, u)
            target_s = self._student_output(
                self.target_nets, x_u, u, s, global_cond=target_obs_cond
            )

        pred_s = self._student_output(self.nets, x_t, t, s, global_cond=obs_cond)

        zero = torch.zeros_like(s)
        pred_0 = self._student_output(
            self.target_nets, pred_s, s, zero, global_cond=target_obs_cond
        )
        with torch.no_grad():
            target_0 = self._student_output(
                self.target_nets, target_s, s, zero, global_cond=target_obs_cond
            )

        weights = self._loss_weights(t)
        return pseudo_huber_loss(
            pred_0,
            target_0.detach(),
            delta=self.algo_config.consistency.loss.delta,
            weights=weights,
        )

    def _dsm_loss(self, batch):
        actions = batch["actions"]
        B = actions.shape[0]
        obs_cond = self._get_obs_cond(batch["obs"], batch["goal_obs"], self.nets)
        t = self._sample_diffusion_timesteps(B, actions.device)
        zero = torch.zeros_like(t)

        noise = torch.randn_like(actions)
        x_t = self.noise_scheduler.add_noise(actions, noise, t)
        pred_0 = self._student_output(self.nets, x_t, t, zero, global_cond=obs_cond)

        weights = self._loss_weights(t)
        return pseudo_huber_loss(
            pred_0,
            actions,
            delta=self.algo_config.consistency.loss.delta,
            weights=weights,
        )

    def _sample_ctm_timesteps(self, batch_size, device):
        if self.algo_config.consistency.time_sampler != "uniform":
            raise ValueError("Only uniform consistency time sampling is implemented.")

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        t = torch.randint(1, num_train_timesteps, (batch_size,), device=device).long()
        s = torch.floor(torch.rand(batch_size, device=device) * t.float()).long()
        teacher_steps = max(1, int(self.algo_config.consistency.teacher_steps))
        u = torch.maximum(t - teacher_steps, s)
        return t, s, u.long()

    def _sample_diffusion_timesteps(self, batch_size, device):
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        return torch.randint(1, num_train_timesteps, (batch_size,), device=device).long()

    def _get_obs_cond(self, obs_dict, goal_dict, nets):
        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        assert obs_features.ndim == 3
        return obs_features.flatten(start_dim=1)

    def _student_output(self, nets, sample, timestep, stop_timestep, global_cond):
        eps = nets["policy"]["ctm_net"](
            sample=sample,
            timestep=timestep,
            stop_timestep=stop_timestep,
            global_cond=global_cond,
        )
        return self._ddim_jump(sample, eps, timestep, stop_timestep)

    def _ddim_jump(self, sample, eps, timestep, target_timestep):
        alphas = self.noise_scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        alpha_t = self._extract(alphas, timestep, sample.shape)
        alpha_target = self._extract(alphas, target_timestep, sample.shape)

        pred_x0 = (sample - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        if self.algo_config.consistency.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        return alpha_target.sqrt() * pred_x0 + (1.0 - alpha_target).sqrt() * eps

    @staticmethod
    def _extract(values, timesteps, broadcast_shape):
        out = values.gather(0, timesteps)
        while out.ndim < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out

    def _loss_weights(self, timesteps):
        if not self.algo_config.consistency.karras_weighting.enabled:
            return None

        alphas = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32
        )
        alpha_t = alphas.gather(0, timesteps)
        sigma = ((1.0 - alpha_t) / alpha_t).sqrt()
        data_std = self.algo_config.consistency.karras_weighting.data_std
        weights = (sigma.pow(2) + data_std**2) / ((sigma * data_std).pow(2) + 1e-8)
        if self.algo_config.consistency.karras_weighting.normalize:
            weights = weights / weights.mean().clamp_min(1e-8)
        return weights

    @torch.no_grad()
    def _update_target_nets(self):
        decay = self.algo_config.consistency.ema_decay
        online_params = [p.data for p in self.nets.parameters()]
        target_params = [p.data for p in self.target_nets.parameters()]
        torch._foreach_mul_(target_params, decay)
        torch._foreach_add_(target_params, online_params, alpha=1.0 - decay)

    def log_info(self, info):
        log = super(ConsistencyPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["loss"].item()
        if "ctm_loss" in info["losses"]:
            log["CTM_Loss"] = info["losses"]["ctm_loss"].item()
        if "dsm_loss" in info["losses"]:
            log["DSM_Loss"] = info["losses"]["dsm_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue = deque(maxlen=self.algo_config.horizon.observation_horizon)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        if len(self.action_queue) == 0:
            action_sequence = self._get_action_trajectory(obs_dict, goal_dict)
            self.action_queue.extend(action_sequence[0])
        return self.action_queue.popleft().unsqueeze(0)

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        policy_nets = self.target_nets
        if not self.algo_config.consistency.use_ema_inference:
            policy_nets = self.nets

        obs_cond = self._get_obs_cond(obs_dict, goal_dict, policy_nets)
        B = obs_cond.shape[0]
        device = self.device
        zero = torch.zeros(B, dtype=torch.long, device=device)
        start_t = torch.full(
            (B,),
            self.noise_scheduler.config.num_train_timesteps - 1,
            dtype=torch.long,
            device=device,
        )

        naction = torch.randn((B, Tp, self.ac_dim), device=device)
        naction = self._student_output(policy_nets, naction, start_t, zero, obs_cond)

        for chain_t in self._inference_chain_timesteps():
            t = torch.full((B,), chain_t, dtype=torch.long, device=device)
            noise = torch.randn_like(naction)
            noisy_action = self.noise_scheduler.add_noise(naction, noise, t)
            naction = self._student_output(policy_nets, noisy_action, t, zero, obs_cond)

        if self.algo_config.consistency.clip_sample:
            naction = naction.clamp(-1.0, 1.0)

        start = To - 1
        end = start + Ta
        return naction[:, start:end]

    def _inference_chain_timesteps(self):
        num_steps = int(self.algo_config.consistency.inference.num_steps)
        if num_steps <= 1:
            return []

        configured = list(self.algo_config.consistency.inference.chaining_timesteps)
        if len(configured) > 0:
            return [int(t) for t in configured[: num_steps - 1]]

        max_t = self.noise_scheduler.config.num_train_timesteps - 1
        return [
            max(1, min(max_t, int(round(max_t * k / num_steps))))
            for k in range(num_steps - 1, 0, -1)
        ]

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "target_nets": self.target_nets.state_dict(),
            "optimizers": {k: self.optimizers[k].state_dict() for k in self.optimizers},
            "lr_schedulers": {
                k: (
                    self.lr_schedulers[k].state_dict()
                    if self.lr_schedulers[k] is not None
                    else None
                )
                for k in self.lr_schedulers
            },
        }

    def deserialize(self, model_dict, load_optimizers=False):
        self.nets.load_state_dict(model_dict["nets"])
        if "target_nets" in model_dict:
            self.target_nets.load_state_dict(model_dict["target_nets"])
        else:
            self.target_nets.load_state_dict(model_dict["nets"])

        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(
                        model_dict["lr_schedulers"][k]
                    )
        elif self.algo_config.consistency.get("drop_teacher_after_deserialize", True):
            self.teacher_nets = None


def pseudo_huber_loss(pred, target, delta=0.0, weights=None):
    if delta == -1:
        delta = math.sqrt(math.prod(pred.shape[1:])) * 0.00054

    diff = pred - target
    if delta == 0:
        loss = diff.pow(2)
    else:
        loss = delta**2 * (torch.sqrt(1.0 + (diff / delta).pow(2)) - 1.0)

    if weights is not None:
        while weights.ndim < loss.ndim:
            weights = weights.unsqueeze(-1)
        loss = loss * weights
    return loss.mean()
