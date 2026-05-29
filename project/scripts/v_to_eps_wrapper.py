"""
v-prediction → epsilon-prediction wrapper for distilled student checkpoints.

After progressive distillation, the student checkpoint ("G" key) is a DPNetwork
that outputs v-prediction internally (DPNetwork wraps ConditionalUnet1D with eps->v).

This module provides:
  - VToEpsPolicy  : robomimic-compatible policy class that loads a distilled student
                    and performs inference using the distiller's GaussianDiffusionDefault
                    (v-prediction denoising). Outputs actions just like DiffusionPolicyUNet.
  - convert_student_to_robomimic()  : convert student.pt back to a robomimic .pth so that
                    eval_one.sh / run_trained_agent.py can be used directly.

v <-> eps conversion:
  Forward (eps -> v):  v = (eps - sigma * z) / alpha
  Inverse (v -> eps):  eps = alpha * v + sigma * z
  x0  from v         : x0 = alpha * z - sigma * v

where alpha = sqrt(alphas_cumprod[t]), sigma = sqrt(1 - alphas_cumprod[t]),
and z is the noisy sample.
"""

import copy
import json
import os
import sys
from collections import OrderedDict, deque

import torch
import torch.nn as nn

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_DISTILLER_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "diffusion_distiller")
_ROBOMIMIC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "robomimic")
for _p in (_SCRIPT_DIR, _DISTILLER_DIR, _ROBOMIMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from v_diffusion import GaussianDiffusionDefault, make_beta_schedule
from dp_module import DPNetwork, _build_networks

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.config import config_factory
from robomimic.algo import register_algo_factory_func, PolicyAlgo


# ============================================================================
#  Conversion helpers
# ============================================================================

def _robomimic_betas(n_ts: int) -> torch.Tensor:
    sched = DDPMScheduler(
        num_train_timesteps=n_ts,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return sched.betas.double()


def _strip_runtime_keys(cfg_dict: dict):
    for net_cfg in cfg_dict.get("algo", {}).get("optim_params", {}).values():
        if isinstance(net_cfg, dict):
            net_cfg.pop("num_train_batches", None)
            net_cfg.pop("num_epochs", None)


def load_student_ckpt(student_pt_path: str, device: torch.device):
    """
    Load a distiller-format student checkpoint.

    Returns:
        dp_net      : DPNetwork (eps->v internally; wraps UNet + obs_encoder)
        diffusion   : GaussianDiffusionDefault (v-prediction denoising)
        cfg         : robomimic config
        shape_meta  : dict with ac_dim and obs key shapes
        n_ts        : int  number of inference timesteps
        time_scale  : float
    """
    ckpt = torch.load(student_pt_path, map_location=device)
    n_ts       = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    cfg_dict   = json.loads(ckpt["robomimic_config"])
    shape_meta = ckpt["shape_metadata"]
    _strip_runtime_keys(cfg_dict)

    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        cfg.update(cfg_dict)

    obs_key_shapes = OrderedDict(
        (k, shape_meta["all_shapes"][k]) for k in shape_meta["all_obs_keys"]
    )
    ac_dim = shape_meta["ac_dim"]

    ObsUtils.initialize_obs_utils_with_config(cfg)
    obs_encoder, unet = _build_networks(
        robomimic_algo_cfg=cfg.algo,
        obs_config=cfg.observation,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
    )
    net = DPNetwork(
        unet=unet,
        obs_encoder=obs_encoder,
        obs_horizon=cfg.algo.horizon.observation_horizon,
        ac_dim=ac_dim,
        num_train_timesteps=100,        # always use the 100-step alpha table
    ).to(device)
    net.load_state_dict(ckpt["G"])
    net.eval()

    betas     = _robomimic_betas(n_ts).to(device)
    diffusion = GaussianDiffusionDefault(net, betas, time_scale=time_scale)
    diffusion.gamma = 0.0

    return net, diffusion, cfg, shape_meta, n_ts, time_scale


# ============================================================================
#  VToEpsPolicy  —  robomimic-compatible inference wrapper
# ============================================================================

class VToEpsPolicy(PolicyAlgo):
    """
    Loads a distilled DPNetwork student and runs inference through
    GaussianDiffusionDefault (v-prediction) to produce actions.

    Compatible with robomimic's RolloutPolicy interface:
      policy.reset()
      action = policy.get_action(obs_dict)
    """

    def _create_networks(self):
        student_pt = self.algo_config.distill.student_ckpt_path
        device = self.device

        self._dp_net, self._diffusion, _, shape_meta, n_ts, time_scale = \
            load_student_ckpt(student_pt, device)

        self._n_ts       = n_ts
        self._time_scale = time_scale
        self._To = self.algo_config.horizon.observation_horizon
        self._Ta = self.algo_config.horizon.action_horizon
        self._Tp = self.algo_config.horizon.prediction_horizon

        # Dummy nets so robomimic's base class doesn't break
        self.nets = nn.ModuleDict()

    # ── mandatory overrides ───────────────────────────────────────────────────

    def process_batch_for_training(self, batch):
        raise NotImplementedError("VToEpsPolicy is inference-only.")

    def train_on_batch(self, batch, epoch, validate=False):
        raise NotImplementedError("VToEpsPolicy is inference-only.")

    def log_info(self, info):
        return {}

    def serialize(self):
        return {"student_ckpt_path": self.algo_config.distill.student_ckpt_path}

    def deserialize(self, model_dict, load_optimizers=False):
        pass  # loaded in _create_networks

    # ── inference ─────────────────────────────────────────────────────────────

    def reset(self):
        self._obs_queue    = deque(maxlen=self._To)
        self._action_queue = deque(maxlen=self._Ta)

    @torch.no_grad()
    def get_action(self, obs_dict, goal_dict=None):
        if len(self._action_queue) == 0:
            seq = self._get_action_trajectory(obs_dict)
            self._action_queue.extend(seq[0])
        return self._action_queue.popleft().unsqueeze(0)

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict):
        net = self._dp_net
        net.eval()

        # ── encode observations ──────────────────────────────────────────────
        inputs = {"obs": obs_dict, "goal": None}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
        obs_features = TensorUtils.time_distributed(
            inputs, net.obs_encoder, inputs_as_kwargs=True
        )
        B = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)   # [B, To*D]

        extra_args = {"global_cond": obs_cond}

        # ── denoising loop (v-prediction via GaussianDiffusionDefault) ───────
        x = torch.randn(B, self._Tp, self._dp_net.ac_dim, device=self.device)
        x = self._diffusion.p_sample_loop(x, extra_args)   # deterministic DDPM

        # ── extract Ta actions ───────────────────────────────────────────────
        start  = self._To - 1
        end    = start + self._Ta
        return x[:, start:end]                             # [B, Ta, Da]


# ============================================================================
#  convert_student_to_robomimic  —  offline conversion for eval_one.sh
# ============================================================================

def convert_student_to_robomimic(student_pt_path: str, out_pth_path: str,
                                  device_str: str = "cpu"):
    """
    Convert a distiller student checkpoint back to robomimic .pth format so
    that eval_one.sh (run_trained_agent.py) can load it directly.

    The converted checkpoint uses 'diffusion_policy' as algo_name and replaces
    the v-predicting UNet with an epsilon-predicting version obtained by inverting
    the eps->v wrapper analytically.

    Because DPNetwork.forward() does: v = (eps_unet - sigma*z) / alpha,
    the underlying UNet still predicts epsilon internally. We can therefore
    re-use the SAME UNet weights directly in a standard DiffusionPolicyUNet —
    no weight modification needed, just repack the state_dict.
    """
    device = torch.device(device_str)
    ckpt = torch.load(student_pt_path, map_location=device)
    n_ts       = ckpt["n_timesteps"]
    cfg_dict   = json.loads(ckpt["robomimic_config"])
    shape_meta = ckpt["shape_metadata"]
    _strip_runtime_keys(cfg_dict)

    # Rewrite config to use DDPM with student's reduced inference steps
    cfg_dict["algo"]["ddpm"]["num_inference_timesteps"] = n_ts
    # Keep algo_name as diffusion_policy (standard robomimic algo)
    # The student's UNet predicts epsilon internally — only the DPNetwork
    # wrapper adds the eps->v conversion, which eval doesn't use.

    # Extract just the inner UNet + obs_encoder weights from DPNetwork state_dict
    full_sd = ckpt["G"]
    nets_sd  = OrderedDict()
    for k, v in full_sd.items():
        if k.startswith("obs_encoder."):
            nets_sd["policy.obs_encoder." + k[len("obs_encoder."):]] = v
        elif k.startswith("unet."):
            nets_sd["policy.noise_pred_net." + k[len("unet."):]] = v
        # skip sqrt_ac, sqrt_1mac (buffers only used for eps->v, not needed in eval)

    robomimic_ckpt = {
        "model": {
            "nets": nets_sd,
            "optimizers": {},
            "lr_schedulers": {},
            "ema": None,
        },
        "config": json.dumps(cfg_dict),
        "algo_name": "diffusion_policy",
        "env_metadata": ckpt.get("env_metadata", {}),
        "shape_metadata": shape_meta,
        "variable_state": {},
        "action_normalization_stats": ckpt.get("action_normalization_stats", {}),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_pth_path)), exist_ok=True)
    torch.save(robomimic_ckpt, out_pth_path)
    print(f"Saved robomimic checkpoint -> {out_pth_path}")
    print(f"  algo: diffusion_policy, num_inference_timesteps={n_ts}")
    print("  NOTE: uses DDIM inference. Set ddpm.enabled=False, ddim.enabled=True")
    print("        with num_inference_timesteps =", n_ts, "for best results.")


# ============================================================================
#  CLI  (convert student.pt -> robomimic .pth)
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Convert distilled student.pt back to robomimic .pth for eval_one.sh"
    )
    p.add_argument("--student_pt",  required=True, help="Distiller-format student (.pt)")
    p.add_argument("--out_pth",     required=True, help="Output robomimic checkpoint (.pth)")
    p.add_argument("--device",      default="cpu")
    args = p.parse_args()
    convert_student_to_robomimic(args.student_pt, args.out_pth, args.device)
