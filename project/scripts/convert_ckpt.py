"""
Convert a robomimic DiffusionPolicy checkpoint (.pth) to the diffusion_distiller format.

Robomimic format:
    {
        "model": {"nets": state_dict, "ema": ema_state_dict, ...},
        "config": json_str,
        "shape_metadata": {...},
        ...
    }

Distiller format (required by distillate.py):
    {
        "G": model_state_dict,   <- DPNetwork weights (obs_encoder + unet)
        "n_timesteps": int,       <- 100
        "time_scale": float,      <- 1.0
    }

Usage:
    python convert_ckpt.py \
        --robomimic_ckpt ../project/ckpt_bundle/lift/last.pth \
        --out            ../project/distill_ckpts/lift_teacher.pt
"""

import argparse
import json
import os
import sys

import torch


def _strip_runtime_keys(cfg_dict: dict):
    """Remove keys added by train.py at runtime (not defined in Config class)."""
    for net_cfg in cfg_dict.get("algo", {}).get("optim_params", {}).values():
        if isinstance(net_cfg, dict):
            net_cfg.pop("num_train_batches", None)
            net_cfg.pop("num_epochs", None)
from collections import OrderedDict

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROBOMIMIC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "robomimic")
if _ROBOMIMIC_DIR not in sys.path:
    sys.path.insert(0, _ROBOMIMIC_DIR)

from diffusers.training_utils import EMAModel
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def convert(robomimic_ckpt_path: str, out_path: str):
    device = torch.device("cpu")
    ckpt = torch.load(robomimic_ckpt_path, map_location=device)

    # ── parse robomimic config from checkpoint ───────────────────────────────
    config_json = ckpt["config"]
    cfg_dict = json.loads(config_json)
    _strip_runtime_keys(cfg_dict)

    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        cfg.update(cfg_dict)

    shape_meta = ckpt["shape_metadata"]
    obs_key_shapes = OrderedDict(
        (k, shape_meta["all_shapes"][k]) for k in shape_meta["all_obs_keys"]
    )
    ac_dim = shape_meta["ac_dim"]

    # ── build DPNetwork (same arch as training) ──────────────────────────────
    ObsUtils.initialize_obs_utils_with_config(cfg)
    sys.path.insert(0, os.path.dirname(__file__))
    from dp_module import DPNetwork, _build_networks

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
        num_train_timesteps=cfg.algo.ddpm.num_train_timesteps,
    ).to(device)

    # ── load weights: prefer EMA (= what's used at inference time) ──────────
    model_dict = ckpt["model"]
    net.obs_encoder.load_state_dict(
        _filter_prefix(model_dict["nets"], "policy.obs_encoder.")
    )
    net.unet.load_state_dict(
        _filter_prefix(model_dict["nets"], "policy.noise_pred_net.")
    )

    if model_dict.get("ema") is not None:
        # Apply EMA to the loaded nets weights
        ema = EMAModel(parameters=_all_params(net.obs_encoder, net.unet),
                       power=cfg.algo.ema.power)
        ema.load_state_dict(model_dict["ema"])
        ema.copy_to(_all_params(net.obs_encoder, net.unet))
        print("EMA weights applied.")
    else:
        print("No EMA found; using raw net weights.")

    # ── save in distiller format ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "G": net.state_dict(),
            "n_timesteps": cfg.algo.ddpm.num_train_timesteps,
            "time_scale": 1.0,
            # store extras for re-loading and back-conversion to robomimic
            "robomimic_config": config_json,
            "shape_metadata": shape_meta,
            "env_metadata": ckpt.get("env_metadata", {}),
            "action_normalization_stats": ckpt.get("action_normalization_stats", {}),
        },
        out_path,
    )
    print(f"Saved distiller checkpoint → {out_path}")
    print(f"  n_timesteps = {cfg.algo.ddpm.num_train_timesteps}, time_scale = 1.0")


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_prefix(state_dict, prefix: str) -> OrderedDict:
    """Extract sub-state-dict with keys starting with prefix, stripping the prefix."""
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _all_params(*modules):
    """Yield all parameters from a sequence of modules (for EMA)."""
    for m in modules:
        yield from m.parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robomimic_ckpt", required=True,
                        help="Path to robomimic .pth checkpoint")
    parser.add_argument("--out", required=True,
                        help="Output path for distiller-format checkpoint (.pt)")
    args = parser.parse_args()
    convert(args.robomimic_ckpt, args.out)
