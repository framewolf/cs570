"""
Robomimic Diffusion Policy adapter for the diffusion_distiller framework.

Provides:
  make_model()    -> DPNetwork  (ConditionalUnet1D wrapped with eps->v conversion)
  make_dataset()  -> torch.utils.data.Dataset  (robomimic HDF5)
  make_condition  -> function(actions, obs_dict, device) -> {"global_cond": tensor}

Usage (via run_distill.py):
  1. Set DP_CONFIG at the bottom before importing this module, or call configure().
  2. convert_ckpt.py converts a robomimic .pth into the distiller format
     {"G": state_dict, "n_timesteps": 100, "time_scale": 1}.
  3. run_distill.py drives distillation using DiffusionDistillation from train_utils.

eps -> v conversion formula (derived from v = alpha*eps - sigma*x0):
  v = (eps_pred - sigma * z) / alpha
where z is the noisy input, alpha = sqrt(alphas_cumprod[t]), sigma = sqrt(1 - alphas_cumprod[t]).
"""

import os
import sys
import math
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ paths
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_DISTILLER_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "diffusion_distiller")
_ROBOMIMIC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "robomimic")
for _p in (_DISTILLER_DIR, _ROBOMIMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.config import config_factory
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ------------------------------------------------------------------ globals
# Filled by configure() before distillation starts.
_GLOBAL_CFG = {}   # keys: hdf5_path, obs_keys, pred_horizon, obs_horizon, ac_dim

# Global obs encoder (teacher's frozen copy) used by make_condition.
_OBS_ENCODER: nn.Module = None


def configure(
    hdf5_path: str,
    obs_keys: list,
    ac_dim: int = 7,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
):
    """Call once before make_model() / make_dataset() / make_condition."""
    _GLOBAL_CFG["hdf5_path"] = hdf5_path
    _GLOBAL_CFG["obs_keys"] = obs_keys
    _GLOBAL_CFG["ac_dim"] = ac_dim
    _GLOBAL_CFG["pred_horizon"] = pred_horizon
    _GLOBAL_CFG["obs_horizon"] = obs_horizon


# ======================================================================
#  Network
# ======================================================================

class DPNetwork(nn.Module):
    """
    Wraps robomimic's ConditionalUnet1D + ObservationGroupEncoder.

    Input  : noisy actions [B, Tp, Da] + global_cond [B, obs_cond_dim]
    Output : v-prediction  [B, Tp, Da]

    v-prediction allows the distillation framework to compute x0 as:
        x0 = alpha * z - sigma * v
    """

    def __init__(self, unet: nn.Module, obs_encoder: nn.Module,
                 obs_horizon: int, ac_dim: int,
                 num_train_timesteps: int = 100):
        super().__init__()
        self.unet = unet
        self.obs_encoder = obs_encoder
        self.obs_horizon = obs_horizon
        self.ac_dim = ac_dim

        # alpha/sigma from the SAME schedule used during robomimic training
        sched = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        ac = sched.alphas_cumprod.float()          # [T]
        self.register_buffer("sqrt_ac",   ac.sqrt())
        self.register_buffer("sqrt_1mac", (1.0 - ac).sqrt())

        # image_size required by distillate.py's make_iter_callback
        self.image_size = [1, num_train_timesteps, ac_dim]

    def forward(self, x: torch.Tensor, t: torch.Tensor, global_cond=None):
        """
        Args:
            x          : [B, Tp, Da]  noisy actions
            t          : [B]  raw timestep indices (after time_scale mult from v_diffusion)
            global_cond: [B, obs_cond_dim]  pre-encoded observation conditioning

        Returns:
            v_pred : [B, Tp, Da]  v-prediction
        """
        t_int = t.long().clamp(0, len(self.sqrt_ac) - 1)

        # epsilon prediction from the ConditionalUnet1D
        eps_pred = self.unet(x.float(), t_int, global_cond=global_cond)

        # eps -> v:  v = (eps - sigma * z) / alpha
        alpha = self.sqrt_ac[t_int].view(-1, 1, 1)
        sigma = self.sqrt_1mac[t_int].view(-1, 1, 1)
        v_pred = (eps_pred - sigma * x) / alpha

        return v_pred.float()


def _build_networks(robomimic_algo_cfg, obs_config, obs_key_shapes: OrderedDict, ac_dim: int):
    """Construct obs_encoder + ConditionalUnet1D from a robomimic config."""
    observation_group_shapes = OrderedDict()
    observation_group_shapes["obs"] = OrderedDict(obs_key_shapes)
    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(obs_config.encoder)

    obs_encoder = ObsNets.ObservationGroupEncoder(
        observation_group_shapes=observation_group_shapes,
        encoder_kwargs=encoder_kwargs,
    )
    obs_dim = obs_encoder.output_shape()[0]

    unet = DPNets.ConditionalUnet1D(
        input_dim=ac_dim,
        global_cond_dim=obs_dim * robomimic_algo_cfg.horizon.observation_horizon,
    )
    return obs_encoder, unet


def make_model(robomimic_cfg=None, obs_key_shapes=None) -> DPNetwork:
    """
    Build a DPNetwork from the robomimic config stored in the teacher checkpoint,
    or from an explicitly provided config.

    If called without args (as distillate.py does), falls back to _GLOBAL_CFG.
    """
    if robomimic_cfg is None:
        # Reconstruct a minimal config from globals (used when called from distillate.py)
        # Requires that configure() was called first.
        raise RuntimeError("Call configure() + use run_distill.py directly.")

    obs_k_shapes = obs_key_shapes or _GLOBAL_CFG.get("obs_key_shapes")

    ObsUtils.initialize_obs_utils_with_config(robomimic_cfg)

    obs_encoder, unet = _build_networks(
        robomimic_algo_cfg=robomimic_cfg.algo,
        obs_config=robomimic_cfg.observation,
        obs_key_shapes=obs_k_shapes,
        ac_dim=robomimic_cfg.algo.horizon.prediction_horizon,  # placeholder; overridden below
    )
    net = DPNetwork(
        unet=unet,
        obs_encoder=obs_encoder,
        obs_horizon=robomimic_cfg.algo.horizon.observation_horizon,
        ac_dim=_GLOBAL_CFG.get("ac_dim", 7),
        num_train_timesteps=robomimic_cfg.algo.ddpm.num_train_timesteps,
    )
    return net


# ======================================================================
#  Dataset
# ======================================================================

class RobomimicHDF5Dataset(torch.utils.data.Dataset):
    """
    Loads (action_sequence, obs_dict) pairs from a robomimic low_dim HDF5 file.

    Returns:
        actions  : [pred_horizon, ac_dim]   normalized to [-1, 1]
        obs_dict : dict {key: [obs_horizon, obs_dim]}  raw observation tensors
    """

    def __init__(self, hdf5_path: str, obs_keys: list,
                 pred_horizon: int = 16, obs_horizon: int = 2):
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        # Build flat index: list of (demo_key, start_idx) pairs
        self._index = []
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in sorted(f["data"].keys()):
                n = f["data"][demo_key]["actions"].shape[0]
                # pad_before = obs_horizon-1 positions (we repeat the first obs)
                for start in range(n):
                    self._index.append((demo_key, start))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        demo_key, start = self._index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            T = demo["actions"].shape[0]

            # --- actions: [pred_horizon, Da], padded at end ---
            a_end = min(start + self.pred_horizon, T)
            actions = demo["actions"][start:a_end]  # (<=pred_horizon, Da)
            if len(actions) < self.pred_horizon:
                pad = np.repeat(actions[-1:], self.pred_horizon - len(actions), axis=0)
                actions = np.concatenate([actions, pad], axis=0)
            actions = torch.from_numpy(actions).float()  # [Tp, Da]

            # --- obs: [obs_horizon, obs_dim] for each key, padded at start ---
            obs_dict = {}
            for k in self.obs_keys:
                o_start = max(0, start - (self.obs_horizon - 1))
                obs_raw = demo["obs"][k][o_start : start + 1]  # up to start inclusive
                # pad at beginning if needed
                if len(obs_raw) < self.obs_horizon:
                    pad = np.repeat(obs_raw[:1], self.obs_horizon - len(obs_raw), axis=0)
                    obs_raw = np.concatenate([pad, obs_raw], axis=0)
                obs_dict[k] = torch.from_numpy(obs_raw).float()  # [To, obs_dim]

        return actions, obs_dict


def make_dataset(hdf5_path: str = None, obs_keys: list = None,
                 pred_horizon: int = None, obs_horizon: int = None):
    """Create the robomimic HDF5 dataset; falls back to _GLOBAL_CFG defaults."""
    return RobomimicHDF5Dataset(
        hdf5_path   = hdf5_path   or _GLOBAL_CFG["hdf5_path"],
        obs_keys    = obs_keys    or _GLOBAL_CFG["obs_keys"],
        pred_horizon= pred_horizon or _GLOBAL_CFG.get("pred_horizon", 16),
        obs_horizon = obs_horizon  or _GLOBAL_CFG.get("obs_horizon", 2),
    )


# ======================================================================
#  Conditioning
# ======================================================================

def set_obs_encoder(encoder: nn.Module):
    """Register the (frozen) teacher obs_encoder for use by make_condition."""
    global _OBS_ENCODER
    _OBS_ENCODER = encoder


@torch.no_grad()
def make_condition(actions, obs_dict, device):
    """
    Encode observations into global conditioning for the UNet.

    Args:
        actions  : [B, Tp, Da]  (unused but required by DiffusionDistillation API)
        obs_dict : dict {key: [B, To, obs_dim]}  raw observation tensors
        device   : torch.device

    Returns:
        {"global_cond": [B, To * obs_dim]}
    """
    assert _OBS_ENCODER is not None, "Call set_obs_encoder() before make_condition."

    # Move to device and ensure float
    obs_on_device = {k: obs_dict[k].to(device).float() for k in obs_dict}
    inputs = {"obs": obs_on_device, "goal": None}

    obs_features = TensorUtils.time_distributed(
        inputs, _OBS_ENCODER, inputs_as_kwargs=True
    )  # [B, To, D]
    obs_cond = obs_features.flatten(start_dim=1)  # [B, To*D]
    return {"global_cond": obs_cond}
