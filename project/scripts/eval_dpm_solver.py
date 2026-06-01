#!/usr/bin/env python3
"""
Evaluate a pre-trained Diffusion Policy checkpoint with DPM-Solver++ at inference time.

The checkpoint was trained with DDPM (100-step). At eval time we hot-swap the noise
scheduler to DPM-Solver++ and run it with fewer steps without any re-training.

Usage:
    python scripts/eval_dpm_solver.py --task lift  --steps 6 --n_rollouts 50
    python scripts/eval_dpm_solver.py --task can   --steps 3 --n_rollouts 50
    python scripts/eval_dpm_solver.py --task square --steps 1 --n_rollouts 50

Environment variables (optional):
    CS570_ROOT     - root of the cs570 workspace (default: ~/cs570-project)
    ROBOMIMIC_DIR  - path to robomimic (default: $CS570_ROOT/robomimic)
    MUJOCO_GL      - rendering backend (default: egl)
"""

import argparse
import json
import sys
import os
import types
from copy import deepcopy

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — support both cs570-project and cs570-feature-DPM-solver layouts
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# Try to locate ROBOMIMIC_DIR
_DEFAULT_CS570_ROOT = os.path.join(os.path.expanduser("~"), "cs570-project")
CS570_ROOT = os.environ.get("CS570_ROOT", _DEFAULT_CS570_ROOT)
ROBOMIMIC_DIR = os.environ.get(
    "ROBOMIMIC_DIR",
    os.path.join(CS570_ROOT, "robomimic"),
)

# Also check the robomimic bundled in this repo
_BUNDLED_ROBOMIMIC = os.path.join(
    os.path.dirname(_PROJECT_DIR), "robomimic"
)

for _rdir in [ROBOMIMIC_DIR, _BUNDLED_ROBOMIMIC]:
    if os.path.isdir(_rdir) and _rdir not in sys.path:
        sys.path.insert(0, _rdir)

from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils


# ---------------------------------------------------------------------------
# Scheduler hot-swap
# ---------------------------------------------------------------------------

def _build_dpm_solver(num_steps: int, solver_order: int = 2) -> DPMSolverMultistepScheduler:
    """Build a DPM-Solver++ scheduler matching the training noise schedule.

    The checkpoint was trained with DDPMScheduler(clip_sample=True), which clips
    the predicted x_0 to [-1, 1] at every denoising step.  DPMSolverMultistepScheduler
    does not expose a clip_sample parameter, so we patch convert_model_output to
    perform the same clipping on the x_0 prediction before it is used by the ODE solver.
    """
    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        algorithm_type="dpmsolver++",
        solver_order=solver_order,
    )

    # Patch convert_model_output to clip predicted x_0 to [-1, 1],
    # matching DDPMScheduler(clip_sample=True, clip_sample_range=1.0).
    _orig_convert = sched.convert_model_output.__func__

    def _convert_with_clip(self, model_output, *args, **kwargs):
        x0_pred = _orig_convert(self, model_output, *args, **kwargs)
        return x0_pred.clamp(-1.0, 1.0)

    sched.convert_model_output = types.MethodType(_convert_with_clip, sched)
    return sched


def patch_policy_with_dpm_solver(rollout_policy, num_steps: int, solver_order: int = 2):
    """
    Hot-swap the noise scheduler inside *rollout_policy* to DPM-Solver++.

    This works for checkpoints trained with any scheduler (DDPM, DDIM, …)
    because the UNet only predicts noise – the solver is independent.

    The instance method ``_get_action_trajectory`` is monkey-patched so that
    it uses *num_steps* regardless of what the saved algo_config says.
    """
    algo = rollout_policy.policy  # DiffusionPolicyUNet instance

    # 1. Replace scheduler object (with clip_sample fix applied inside)
    algo.noise_scheduler = _build_dpm_solver(num_steps, solver_order)

    # 2. Monkey-patch _get_action_trajectory at instance level
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim

        # Use the overridden step count
        _num_inference_timesteps = num_steps

        nets = self.nets
        if self.ema is not None:
            self.ema.store(self.nets.parameters())
            self.ema.copy_to(self.nets.parameters())

        # Encode observations
        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)

        # Sample initial noise
        noisy_action = torch.randn((B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # Denoising loop
        self.noise_scheduler.set_timesteps(_num_inference_timesteps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        if self.ema is not None:
            self.ema.restore(self.nets.parameters())

        start = To - 1
        end = start + Ta
        return naction[:, start:end]

    algo._get_action_trajectory = types.MethodType(_get_action_trajectory, algo)


# ---------------------------------------------------------------------------
# Rollout helpers (mirrors run_trained_agent.py, no video overhead)
# ---------------------------------------------------------------------------

def _run_rollout(policy, env, horizon: int) -> dict:
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    success = False
    step_i = 0

    for step_i in range(horizon):
        act = policy(ob=obs)
        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = env.is_success()["task"]
        obs = deepcopy(next_obs)
        if done or success:
            break

    return {
        "Return": float(total_reward),
        "Horizon": float(step_i + 1),
        "Success_Rate": float(success),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_ckpt(task: str, ckpt_dir: str | None) -> str:
    candidates = []
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, task, "last.pth"))
    # project-local ckpt_bundle
    candidates.append(os.path.join(_PROJECT_DIR, "ckpt_bundle", task, "last.pth"))
    # cs570-project ckpt_bundle
    candidates.append(
        os.path.join(
            os.path.expanduser("~"), "cs570-project", "project", "ckpt_bundle", task, "last.pth"
        )
    )
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Checkpoint for task '{task}' not found. Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Diffusion Policy with DPM-Solver++ at N inference steps"
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["lift", "can", "square", "transport", "tool_hang"],
        help="robosuite task name"
    )
    parser.add_argument(
        "--steps", type=int, required=True, choices=[1, 3, 6],
        help="number of DPM-Solver++ inference steps"
    )
    parser.add_argument(
        "--n_rollouts", type=int, default=50,
        help="number of evaluation rollouts (default: 50)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default: 0)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="directory containing <task>/last.pth (auto-detected if omitted)"
    )
    parser.add_argument(
        "--solver_order", type=int, default=2,
        help="DPM-Solver order: 1 or 2 (default: 2). For --steps 1 automatically uses order=1"
    )
    args = parser.parse_args()

    # For 1-step inference, DPM-Solver must use order=1
    solver_order = 1 if args.steps == 1 else args.solver_order

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")

    ckpt_path = _find_ckpt(args.task, args.ckpt_dir)

    print("=" * 60)
    print(f"Task        : {args.task}")
    print(f"Scheduler   : DPM-Solver++ ({args.steps} steps, order={solver_order})")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Rollouts    : {args.n_rollouts}")
    print(f"GPU         : {args.gpu}")
    print("=" * 60)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy from checkpoint (scheduler embedded in ckpt is irrelevant)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )

    # Hot-swap to DPM-Solver++
    patch_policy_with_dpm_solver(policy, num_steps=args.steps, solver_order=solver_order)
    print(f"\nScheduler replaced with DPM-Solver++ ({args.steps} steps, order={solver_order})\n")

    # Build environment from checkpoint metadata
    horizon_map = {"lift": 400, "can": 400, "square": 400, "transport": 700, "tool_hang": 700}
    horizon = horizon_map[args.task]

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=True,
    )

    # Deterministic seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Run rollouts
    rollout_stats = []
    for i in range(args.n_rollouts):
        stats = _run_rollout(policy, env, horizon)
        rollout_stats.append(stats)
        print(
            f"  [{i+1:3d}/{args.n_rollouts}] "
            f"success={int(stats['Success_Rate'])}  "
            f"horizon={int(stats['Horizon']):4d}  "
            f"return={stats['Return']:.2f}",
            flush=True,
        )

    # Aggregate
    avg = {k: float(np.mean([s[k] for s in rollout_stats])) for k in rollout_stats[0]}
    avg["Num_Success"] = float(sum(s["Success_Rate"] for s in rollout_stats))
    avg["success_time_sec"] = avg["Horizon"] / 20.0

    print(f"\n=== {args.task} | DPM-Solver++ {args.steps}-step ===")
    print(json.dumps(avg, indent=2))

    # Save JSON result
    out_dir = os.path.join(_PROJECT_DIR, "outputs", "eval_dpm")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.task}_dpm{args.steps}.json")
    result = {
        "task": args.task,
        "scheduler": "dpmsolver++",
        "steps": args.steps,
        "solver_order": solver_order,
        "n_rollouts": args.n_rollouts,
        **avg,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved → {out_path}")


if __name__ == "__main__":
    main()
