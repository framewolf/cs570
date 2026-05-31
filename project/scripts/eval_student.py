"""
Evaluate a distilled student checkpoint using CORRECT v-prediction inference.

Standard eval_one.sh / run_trained_agent.py treats the UNet as an epsilon predictor,
which is WRONG after progressive distillation. The distilled UNet outputs a modified
quantity that must go through GaussianDiffusionDefault (v-prediction denoising).

This script:
  1. Loads student.pt (distiller format)
  2. Creates DPNetwork + GaussianDiffusionDefault
  3. Runs rollouts with v-prediction denoising loop
  4. Reports success rate (same format as run_trained_agent.py)

Usage:
    python eval_student.py \\
        --student_pt  ../../project/distill_ckpts/lift/student_step001/student.pt \\
        --task        lift \\
        --n_rollouts  50 \\
        --horizon     400 \\
        --video_path  ../../project/outputs/eval/lift_1step.mp4
"""

import argparse
import json
import os
import sys
from collections import OrderedDict, deque
from copy import deepcopy

import imageio
import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_DISTILLER_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "diffusion_distiller")
_ROBOMIMIC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "robomimic")
for _p in (_SCRIPT_DIR, _DISTILLER_DIR, _ROBOMIMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase

from v_to_eps_wrapper import load_student_ckpt


# ============================================================================
#  Policy wrapper
# ============================================================================

class StudentPolicy:
    """
    Wraps a distilled DPNetwork for rollout execution.
    Uses GaussianDiffusionDefault.p_sample_loop for v-prediction inference.
    """

    def __init__(self, dp_net, diffusion, cfg, device):
        self.dp_net    = dp_net
        self.diffusion = diffusion
        self.device    = device
        self.To = cfg.algo.horizon.observation_horizon
        self.Ta = cfg.algo.horizon.action_horizon
        self.Tp = cfg.algo.horizon.prediction_horizon
        self.ac_dim = dp_net.ac_dim

        self._obs_queue    = deque(maxlen=self.To)
        self._action_queue = deque(maxlen=self.Ta)

    def start_episode(self):
        self._obs_queue.clear()
        self._action_queue.clear()

    def __call__(self, ob):
        """
        ob: dict {key: np.array [obs_dim]}  (single timestep, from env)
        Returns: np.array [ac_dim]
        """
        # Push observation to queue
        self._obs_queue.append(ob)

        if len(self._action_queue) == 0:
            # Pad obs queue to To if not full yet
            while len(self._obs_queue) < self.To:
                self._obs_queue.appendleft(deepcopy(ob))

            actions = self._get_action_trajectory()
            self._action_queue.extend(actions)

        return self._action_queue.popleft()

    @torch.no_grad()
    def _get_action_trajectory(self):
        self.dp_net.eval()

        # Stack To observations → [1, To, obs_dim] per key
        obs_stacked = {}
        obs_list = list(self._obs_queue)
        for key in obs_list[0].keys():
            arr = np.stack([o[key] for o in obs_list], axis=0)   # [To, D]
            obs_stacked[key] = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)  # [1, To, D]

        # Encode observations with the teacher's obs_encoder (frozen)
        inputs = {"obs": obs_stacked, "goal": None}
        obs_features = TensorUtils.time_distributed(
            inputs, self.dp_net.obs_encoder, inputs_as_kwargs=True
        )  # [1, To, D]
        obs_cond = obs_features.flatten(start_dim=1)   # [1, To*D]
        extra_args = {"global_cond": obs_cond}

        # v-prediction denoising loop via GaussianDiffusionDefault
        x = torch.randn(1, self.Tp, self.ac_dim, device=self.device)
        x = self.diffusion.p_sample_loop(x, extra_args)   # [1, Tp, Da]

        # Extract Ta actions starting from To-1
        start = self.To - 1
        end   = start + self.Ta
        actions = x[0, start:end].cpu().numpy()  # [Ta, Da]
        return list(actions)  # list of [Da] arrays


# ============================================================================
#  Rollout
# ============================================================================

def run_rollout(policy, env, horizon, video_writer=None, video_skip=5):
    policy.start_episode()
    obs  = env.reset()
    state_dict = env.get_state()
    obs  = env.reset_to(state_dict)

    total_reward = 0.0
    success      = False
    for step_i in range(horizon):
        act  = policy(ob=obs)
        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = env.is_success()["task"]

        if video_writer is not None and step_i % video_skip == 0:
            img = env.render(mode="rgb_array", height=512, width=512,
                             camera_name="agentview")
            video_writer.append_data(img)

        if done or success:
            break
        obs = deepcopy(next_obs)

    return {"Return": total_reward, "Horizon": step_i + 1, "Success_Rate": float(success)}


# ============================================================================
#  Main
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load student ─────────────────────────────────────────────────────────
    dp_net, diffusion, cfg, shape_meta, n_ts, time_scale = load_student_ckpt(
        args.student_pt, device
    )
    dp_net.eval()
    print(f"Loaded student: n_timesteps={n_ts}, time_scale={time_scale}")

    # ── build environment from env_metadata ──────────────────────────────────
    ckpt = torch.load(args.student_pt, map_location="cpu")
    env_meta = ckpt.get("env_metadata", {})

    # student.pt may not carry env_metadata — pull from original robomimic ckpt
    if (not env_meta or "env_name" not in env_meta) and args.orig_ckpt:
        orig = torch.load(args.orig_ckpt, map_location="cpu")
        env_meta = orig.get("env_metadata", {})

    ObsUtils.initialize_obs_utils_with_config(cfg)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=(args.video_path is not None),
    )
    print(f"Environment: {env_meta.get('env_name')}")

    # ── policy ────────────────────────────────────────────────────────────────
    policy = StudentPolicy(dp_net, diffusion, cfg, device)

    # ── rollouts ──────────────────────────────────────────────────────────────
    video_writer = None
    if args.video_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)

    horizon = args.horizon
    if horizon is None:
        case = env_meta.get("env_name", "").lower()
        horizon = 700 if any(t in case for t in ["transport", "tool_hang"]) else 400

    print(f"Running {args.n_rollouts} rollouts (horizon={horizon}) ...")
    stats_list = []
    for i in range(args.n_rollouts):
        s = run_rollout(policy, env, horizon, video_writer, video_skip=5)
        stats_list.append(s)
        print(f"  [{i+1:3d}/{args.n_rollouts}] success={s['Success_Rate']:.0f}  "
              f"horizon={s['Horizon']:3d}  return={s['Return']:.2f}")

    if video_writer:
        video_writer.close()
        print(f"Video saved to: {args.video_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    avg = {k: np.mean([s[k] for s in stats_list]) for k in stats_list[0]}
    avg["Num_Success"] = sum(s["Success_Rate"] for s in stats_list)
    print("\nAverage Rollout Stats")
    print(json.dumps(avg, indent=4))
    return avg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--student_pt",  required=True, help="distiller-format student.pt")
    p.add_argument("--orig_ckpt",   default=None,  help="original robomimic .pth to pull env_metadata")
    p.add_argument("--task",        default=None,  help="task name (for env fallback)")
    p.add_argument("--n_rollouts",  type=int, default=50)
    p.add_argument("--horizon",     type=int, default=None)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--video_path",  default=None)
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    main(args)
