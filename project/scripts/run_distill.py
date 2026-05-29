"""
One round of Progressive Distillation for Robomimic Diffusion Policy.

Pipeline:
  convert_ckpt.py  ->  teacher.pt  (distiller format)
  run_distill.py   ->  student.pt  (halved n_timesteps)

Repeat run_distill.py with the previous student as teacher:
  100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 step

Usage:
  # Round 1: 100 -> 50 steps
  python run_distill.py \\
      --teacher_ckpt ../../project/distill_ckpts/lift_teacher.pt \\
      --out_ckpt     ../../project/distill_ckpts/lift_50.pt \\
      --hdf5_path    ../../robomimic/datasets/lift/ph/low_dim_v15.hdf5

  # Round 2: 50 -> 25 steps
  python run_distill.py \\
      --teacher_ckpt ../../project/distill_ckpts/lift_50.pt \\
      --out_ckpt     ../../project/distill_ckpts/lift_25.pt \\
      --hdf5_path    ../../robomimic/datasets/lift/ph/low_dim_v15.hdf5
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR    = os.path.join(_SCRIPT_DIR, "..")
_DISTILLER_DIR  = os.path.join(_PROJECT_DIR, "..", "diffusion_distiller")
_ROBOMIMIC_DIR  = os.path.join(_PROJECT_DIR, "..", "robomimic")

for _p in (_SCRIPT_DIR, _DISTILLER_DIR, _ROBOMIMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v_diffusion import GaussianDiffusionDefault, make_beta_schedule
from train_utils import DiffusionDistillation, InfinityDataset
from moving_average import init_ema_model, moving_average
from strategies import StrategyConstantLR
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def _strip_runtime_keys(cfg_dict: dict):
    """Remove train.py runtime keys (num_train_batches, num_epochs) from optim_params."""
    for net_cfg in cfg_dict.get("algo", {}).get("optim_params", {}).values():
        if isinstance(net_cfg, dict):
            net_cfg.pop("num_train_batches", None)
            net_cfg.pop("num_epochs", None)
from dp_module import (
    DPNetwork, _build_networks,
    make_dataset, make_condition, set_obs_encoder,
)


# ============================================================================
#  Helpers
# ============================================================================

def _robomimic_betas(n_timesteps: int) -> torch.Tensor:
    """Betas matching the squaredcos_cap_v2 DDPMScheduler used during robomimic training."""
    sched = DDPMScheduler(
        num_train_timesteps=n_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return sched.betas.double()


def _make_diffusion(net: nn.Module, n_ts: int, time_scale: float,
                    gamma: float, device: torch.device) -> GaussianDiffusionDefault:
    betas = _robomimic_betas(n_ts).to(device)
    diff = GaussianDiffusionDefault(net, betas, time_scale=time_scale)
    diff.gamma = gamma
    return diff


def _load_teacher_ckpt(path: str, device: torch.device):
    """
    Load a distiller-format checkpoint.
    Returns (net, cfg, cfg_json, shape_meta, env_meta, n_timesteps, time_scale).
    """
    ckpt = torch.load(path, map_location=device)
    n_ts       = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    cfg_json   = ckpt["robomimic_config"]
    shape_meta = ckpt["shape_metadata"]
    env_meta   = ckpt.get("env_metadata", {})

    cfg_dict = json.loads(cfg_json)
    _strip_runtime_keys(cfg_dict)
    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        cfg.update(cfg_dict)

    obs_key_shapes = OrderedDict(
        (k, shape_meta["all_shapes"][k]) for k in shape_meta["all_obs_keys"]
    )
    ac_dim = shape_meta["ac_dim"]

    ObsUtils.initialize_obs_utils_with_config(cfg)
    # DPNetwork's alpha table always uses the ORIGINAL 100-step training schedule,
    # regardless of the distillation level's inference step count.
    n_ts_orig = cfg.algo.ddpm.num_train_timesteps  # always 100
    net = _build_dpnet(cfg, obs_key_shapes, ac_dim, n_ts_orig).to(device)
    net.load_state_dict(ckpt["G"])
    return net, cfg, cfg_json, shape_meta, env_meta, n_ts, time_scale


def _build_dpnet(cfg, obs_key_shapes: OrderedDict, ac_dim: int, n_ts: int) -> DPNetwork:
    """Construct a fresh DPNetwork from a robomimic config."""
    obs_encoder, unet = _build_networks(
        robomimic_algo_cfg=cfg.algo,
        obs_config=cfg.observation,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
    )
    return DPNetwork(
        unet=unet,
        obs_encoder=obs_encoder,
        obs_horizon=cfg.algo.horizon.observation_horizon,
        ac_dim=ac_dim,
        num_train_timesteps=n_ts,
    )


def _save_student(path: str, net: nn.Module, n_ts: int, time_scale: float,
                  cfg_json: str, shape_meta: dict, env_meta: dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "G": net.state_dict(),
            "n_timesteps": n_ts,
            "time_scale": time_scale,
            "robomimic_config": cfg_json,
            "shape_metadata": shape_meta,
            "env_metadata": env_meta or {},
        },
        path,
    )
    print(f"  Saved -> {path}  (n_timesteps={n_ts}, time_scale={time_scale})")


# ============================================================================
#  Collate function for (actions, obs_dict) batches
# ============================================================================

def _collate_fn(batch):
    actions  = torch.stack([b[0] for b in batch], dim=0)
    obs_keys = list(batch[0][1].keys())
    obs_dict = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in obs_keys}
    return actions, obs_dict


# ============================================================================
#  Main
# ============================================================================

def run_distillation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── teacher ──────────────────────────────────────────────────────────────
    print(f"\nLoading teacher: {args.teacher_ckpt}")
    teacher_net, cfg, cfg_json, shape_meta, env_meta, n_ts_t, ts_t = _load_teacher_ckpt(
        args.teacher_ckpt, device
    )
    teacher_net.eval()
    for p in teacher_net.parameters():
        p.requires_grad_(False)

    # teacher obs_encoder is frozen -> used by make_condition
    set_obs_encoder(teacher_net.obs_encoder)
    print(f"  n_timesteps={n_ts_t}, time_scale={ts_t}")

    teacher_diffusion = _make_diffusion(teacher_net, n_ts_t, ts_t, args.gamma, device)

    # ── student ───────────────────────────────────────────────────────────────
    n_ts_s  = max(1, n_ts_t // 2)
    ts_s    = ts_t * 2.0
    obs_key_shapes = OrderedDict(
        (k, shape_meta["all_shapes"][k]) for k in shape_meta["all_obs_keys"]
    )
    ac_dim = shape_meta["ac_dim"]
    print(f"  student: n_timesteps={n_ts_s}, time_scale={ts_s}")

    n_ts_orig = cfg.algo.ddpm.num_train_timesteps  # always 100 — alpha table never changes
    student_net = _build_dpnet(cfg, obs_key_shapes, ac_dim, n_ts_orig).to(device)
    student_ema = _build_dpnet(cfg, obs_key_shapes, ac_dim, n_ts_orig).to(device)
    init_ema_model(teacher_net, student_net, device)
    init_ema_model(teacher_net, student_ema, device)
    print("  Student weights initialized from teacher.")

    student_diffusion     = _make_diffusion(student_net, n_ts_s, ts_s, args.gamma, device)
    student_ema_diffusion = _make_diffusion(student_ema, n_ts_s, ts_s, args.gamma, device)

    # ── dataset ───────────────────────────────────────────────────────────────
    base_ds = make_dataset(
        hdf5_path=args.hdf5_path,
        obs_keys=list(shape_meta["all_obs_keys"]),
        pred_horizon=cfg.algo.horizon.prediction_horizon,
        obs_horizon=cfg.algo.horizon.observation_horizon,
    )
    ds     = InfinityDataset(base_ds, args.num_iters)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate_fn,
    )

    # ── wandb init ────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and args.wandb_project
    if use_wandb:
        # run name follows the format: e.g. "lift_student_012step"
        task_tag = os.path.basename(args.hdf5_path).replace(".hdf5", "")  # low_dim_v15
        hdf5_parts = args.hdf5_path.split(os.sep)
        task_name = hdf5_parts[-4] if len(hdf5_parts) >= 4 else "task"   # lift / can / ...
        run_name  = f"{task_name}_student_{n_ts_s:03d}step"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "task":         task_name,
                "n_ts_teacher": n_ts_t,
                "n_ts_student": n_ts_s,
                "time_scale_student": ts_s,
                "num_iters":    args.num_iters,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "gamma":        args.gamma,
                "save_every":   args.save_every,
            },
        )
        print(f"wandb run: {wandb.run.url}")

    # ── distillation loop ─────────────────────────────────────────────────────
    strategy = StrategyConstantLR()
    strategy.init(student_diffusion, args.lr, args.num_iters)
    teacher_diffusion.net_.eval()
    student_diffusion.net_.train()

    save_every = args.save_every
    out_dir    = os.path.dirname(os.path.abspath(args.out_ckpt))

    print(f"\nDistilling for {args.num_iters} iterations  (save every {save_every}) ...")
    pbar    = tqdm(loader, total=args.num_iters)
    N       = 0
    L_tot   = 0.0
    L_win   = 0.0   # windowed loss for display

    for actions, obs_dict in pbar:
        strategy.zero_grad()
        actions = actions.to(device)
        B = actions.shape[0]

        # timestep sampling: even indices in teacher's range
        time = 2 * torch.randint(0, student_diffusion.num_timesteps, (B,), device=device)

        extra_args = make_condition(actions, obs_dict, device)
        loss = teacher_diffusion.distill_loss(student_diffusion, actions, time, extra_args)

        loss.backward()
        nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
        strategy.step()
        moving_average(student_net, student_ema)

        N     += 1
        L_tot += loss.item()
        L_win += loss.item()
        pbar.set_description(f"loss={L_tot/N:.6f}")

        # ── wandb step log ────────────────────────────────────────────────
        if use_wandb:
            wandb.log({"distill_loss": loss.item(), "iter": N})

        # ── periodic checkpoint (each in its own subfolder) ───────────────
        if save_every > 0 and N % save_every == 0:
            ckpt_folder = os.path.join(out_dir, f"iter_{N:07d}")
            os.makedirs(ckpt_folder, exist_ok=True)
            mid_path = os.path.join(ckpt_folder, "student.pt")
            _save_student(mid_path, student_ema, n_ts_s, ts_s, cfg_json, shape_meta, env_meta)
            avg_win = L_win / save_every
            print(f"  [iter {N}] avg_loss={avg_win:.6f}  -> {ckpt_folder}/")
            if use_wandb:
                wandb.log({"avg_loss_window": avg_win, "iter": N})
            L_win = 0.0

        if strategy.stop(N, args.num_iters):
            break

    # ── final save ────────────────────────────────────────────────────────────
    final_loss = L_tot / max(N, 1)
    print(f"\nFinal avg loss: {final_loss:.6f}")
    if use_wandb:
        wandb.log({"final_avg_loss": final_loss})
        wandb.finish()
    # EMA weights (student.pt)
    _save_student(args.out_ckpt, student_ema, n_ts_s, ts_s, cfg_json, shape_meta, env_meta)
    # Raw weights (student_raw.pt) — saved alongside EMA for ablation
    raw_path = args.out_ckpt.replace("student.pt", "student_raw.pt")
    _save_student(raw_path, student_net, n_ts_s, ts_s, cfg_json, shape_meta, env_meta)
    print(f"  Raw weights -> {raw_path}")


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--out_ckpt",     required=True)
    p.add_argument("--hdf5_path",    required=True)
    p.add_argument("--num_iters",    type=int,   default=10000)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--gamma",        type=float, default=0.0,
                   help="SNR weighting gamma (0 = no weighting, paper default)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_every",   type=int,   default=0,
                   help="Save intermediate checkpoint every N iters (0 = final only)")
    p.add_argument("--wandb_project", type=str, default="diffusion-policy-pd",
                   help="wandb project name (set empty string to disable)")
    args = p.parse_args()
    run_distillation(args)
