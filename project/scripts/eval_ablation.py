"""
Ablation study:
  A. teacher 50-step rollout   (original epsilon model, DDIM 50 steps)
  B. teacher 25-step rollout   (original epsilon model, DDIM 25 steps)
  C. student 25-step rollout with raw weights
  D. student 25-step rollout with EMA weights

Usage:
  python eval_ablation.py \\
      --teacher_ckpt  ../../ckpt_bundle/lift/last.pth \\
      --student_ema   ../../distill_ckpts/lift/student_step025/student.pt \\
      --student_raw   ../../distill_ckpts/lift/student_step025/student_raw.pt \\
      --n_rollouts    20 \\
      --horizon       400
"""

import argparse
import json
import os
import sys
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_DISTILLER_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "diffusion_distiller")
_ROBOMIMIC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "robomimic")
for _p in (_SCRIPT_DIR, _DISTILLER_DIR, _ROBOMIMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import RolloutPolicy

from v_to_eps_wrapper import load_student_ckpt, _strip_runtime_keys
from dp_module import _build_networks, DPNetwork


# ============================================================================
#  Teacher epsilon-prediction policy (A & B)
# ============================================================================

class TeacherDDIMPolicy:
    """
    Loads the original robomimic epsilon-prediction teacher and runs
    inference with DDIM at a reduced number of steps.
    No distillation — pure inference-time speedup.
    """

    def __init__(self, robomimic_ckpt_path, n_ddim_steps, device):
        import json
        from robomimic.config import config_factory
        from collections import OrderedDict

        self.device = device
        self.n_ddim_steps = n_ddim_steps

        ckpt = torch.load(robomimic_ckpt_path, map_location=device)
        cfg_dict = json.loads(ckpt["config"])
        _strip_runtime_keys(cfg_dict)
        shape_meta = ckpt["shape_metadata"]

        cfg = config_factory(cfg_dict["algo_name"])
        with cfg.values_unlocked():
            cfg.update(cfg_dict)

        obs_key_shapes = {k: shape_meta["all_shapes"][k] for k in shape_meta["all_obs_keys"]}
        ac_dim = shape_meta["ac_dim"]

        ObsUtils.initialize_obs_utils_with_config(cfg)

        # Build networks
        from collections import OrderedDict as OD
        obs_encoder, unet = _build_networks(
            cfg.algo, cfg.observation, OD(obs_key_shapes), ac_dim
        )

        # Load weights directly (no EMA applied — teacher uses raw nets)
        nets_sd = ckpt["model"]["nets"]
        obs_enc_sd = {k[len("policy.obs_encoder."):]: v
                      for k, v in nets_sd.items() if k.startswith("policy.obs_encoder.")}
        unet_sd    = {k[len("policy.noise_pred_net."):]: v
                      for k, v in nets_sd.items() if k.startswith("policy.noise_pred_net.")}

        # Apply EMA if available
        from diffusers.training_utils import EMAModel
        if ckpt["model"].get("ema") is not None:
            obs_encoder.load_state_dict(obs_enc_sd)
            unet.load_state_dict(unet_sd)
            all_params = list(obs_encoder.parameters()) + list(unet.parameters())
            ema = EMAModel(parameters=all_params, power=cfg.algo.ema.power)
            ema.load_state_dict(ckpt["model"]["ema"])
            ema.copy_to(all_params)
        else:
            obs_encoder.load_state_dict(obs_enc_sd)
            unet.load_state_dict(unet_sd)

        self.obs_encoder = obs_encoder.to(device).eval()
        self.unet        = unet.to(device).eval()

        # DDIM scheduler (deterministic, fewer steps)
        self.ddim = DDIMScheduler(
            num_train_timesteps=cfg.algo.ddpm.num_train_timesteps,
            beta_schedule=cfg.algo.ddpm.beta_schedule,
            clip_sample=cfg.algo.ddpm.clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        self.To     = cfg.algo.horizon.observation_horizon
        self.Ta     = cfg.algo.horizon.action_horizon
        self.Tp     = cfg.algo.horizon.prediction_horizon
        self.ac_dim = ac_dim

        self._obs_queue    = deque(maxlen=self.To)
        self._action_queue = deque(maxlen=self.Ta)

    def start_episode(self):
        self._obs_queue.clear()
        self._action_queue.clear()

    def __call__(self, ob):
        self._obs_queue.append(ob)
        if len(self._action_queue) == 0:
            while len(self._obs_queue) < self.To:
                self._obs_queue.appendleft(deepcopy(ob))
            actions = self._get_actions()
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    @torch.no_grad()
    def _get_actions(self):
        obs_stacked = {}
        obs_list = list(self._obs_queue)
        for key in obs_list[0]:
            arr = np.stack([o[key] for o in obs_list], axis=0)
            obs_stacked[key] = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)

        inputs = {"obs": obs_stacked, "goal": None}
        feats = TensorUtils.time_distributed(inputs, self.obs_encoder, inputs_as_kwargs=True)
        obs_cond = feats.flatten(start_dim=1)   # [1, To*D]

        # DDIM epsilon-prediction denoising
        naction = torch.randn(1, self.Tp, self.ac_dim, device=self.device)
        self.ddim.set_timesteps(self.n_ddim_steps)
        for t in self.ddim.timesteps:
            eps = self.unet(naction, t, global_cond=obs_cond)
            naction = self.ddim.step(eps, t, naction).prev_sample

        start  = self.To - 1
        return list(naction[0, start:start + self.Ta].cpu().numpy())


# ============================================================================
#  Rollout runner
# ============================================================================

def run_rollout(policy, env, horizon):
    policy.start_episode()
    obs   = env.reset()
    state = env.get_state()
    obs   = env.reset_to(state)
    total_r, success = 0.0, False
    for step_i in range(horizon):
        act  = policy(ob=obs)
        obs, r, done, _ = env.step(act)
        total_r += r
        success  = env.is_success()["task"]
        if done or success:
            break
    return {"Return": total_r, "Horizon": step_i + 1, "Success_Rate": float(success)}


def run_eval(label, policy, env, n_rollouts, horizon):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    stats = []
    for i in range(n_rollouts):
        s = run_rollout(policy, env, horizon)
        stats.append(s)
        print(f"  [{i+1:3d}/{n_rollouts}] success={s['Success_Rate']:.0f}  "
              f"horizon={s['Horizon']:3d}")
    avg = {k: np.mean([s[k] for s in stats]) for k in stats[0]}
    avg["Num_Success"] = sum(s["Success_Rate"] for s in stats)
    print(f"  → Success_Rate={avg['Success_Rate']:.2f}  ({int(avg['Num_Success'])}/{n_rollouts})")
    return avg


# ============================================================================
#  Student v-prediction policy factory
# ============================================================================

def make_student_policy(student_pt_path, device, cfg, obs_key_shapes, ac_dim, n_ts_orig):
    """Load a student.pt and return a StudentPolicy-like object."""
    from project_scripts_eval_student import StudentPolicy  # local import trick
    dp_net, diffusion, cfg_s, shape_meta, n_ts, ts = load_student_ckpt(
        student_pt_path, device
    )
    dp_net.eval()
    return dp_net, diffusion, cfg_s


# avoid circular import — inline the StudentPolicy here
class StudentPolicy:
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
        self._obs_queue.append(ob)
        if len(self._action_queue) == 0:
            while len(self._obs_queue) < self.To:
                self._obs_queue.appendleft(deepcopy(ob))
            self._action_queue.extend(self._get_actions())
        return self._action_queue.popleft()

    @torch.no_grad()
    def _get_actions(self):
        self.dp_net.eval()
        obs_list = list(self._obs_queue)
        obs_stacked = {}
        for key in obs_list[0]:
            arr = np.stack([o[key] for o in obs_list], axis=0)
            obs_stacked[key] = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)
        inputs = {"obs": obs_stacked, "goal": None}
        feats = TensorUtils.time_distributed(inputs, self.dp_net.obs_encoder, inputs_as_kwargs=True)
        obs_cond = feats.flatten(start_dim=1)
        x = torch.randn(1, self.Tp, self.dp_net.ac_dim, device=self.device)
        x = self.diffusion.p_sample_loop(x, {"global_cond": obs_cond})
        start = self.To - 1
        return list(x[0, start:start + self.Ta].cpu().numpy())


# ============================================================================
#  Main
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── build env from teacher ckpt ──────────────────────────────────────────
    orig_ck   = torch.load(args.teacher_ckpt, map_location="cpu")
    env_meta  = orig_ck.get("env_metadata", {})
    ObsUtils.initialize_obs_utils_with_config(
        __import__("robomimic.config", fromlist=["config_factory"]).config_factory(
            json.loads(orig_ck["config"])["algo_name"]
        )
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False
    )
    print(f"Environment: {env_meta.get('env_name')}")

    horizon = args.horizon or (700 if any(t in env_meta.get("env_name","").lower()
                                          for t in ["transport","tool_hang"]) else 400)
    results = {}

    # ── A: teacher 50-step DDIM ──────────────────────────────────────────────
    if args.teacher_ckpt:
        policy_A = TeacherDDIMPolicy(args.teacher_ckpt, n_ddim_steps=50, device=device)
        results["A_teacher_50step"] = run_eval(
            "A. Teacher — 50-step DDIM (inference-time reduction, no distillation)",
            policy_A, env, args.n_rollouts, horizon
        )

    # ── B: teacher 25-step DDIM ──────────────────────────────────────────────
    if args.teacher_ckpt:
        policy_B = TeacherDDIMPolicy(args.teacher_ckpt, n_ddim_steps=25, device=device)
        results["B_teacher_25step"] = run_eval(
            "B. Teacher — 25-step DDIM (inference-time reduction, no distillation)",
            policy_B, env, args.n_rollouts, horizon
        )

    # ── C: student 25-step raw weights ───────────────────────────────────────
    if args.student_raw:
        dp_c, diff_c, cfg_c, *_ = load_student_ckpt(args.student_raw, device)
        policy_C = StudentPolicy(dp_c, diff_c, cfg_c, device)
        results["C_student_25step_raw"] = run_eval(
            "C. Student 25-step — RAW weights (no EMA)",
            policy_C, env, args.n_rollouts, horizon
        )

    # ── D: student 25-step EMA weights ───────────────────────────────────────
    if args.student_ema:
        dp_d, diff_d, cfg_d, *_ = load_student_ckpt(args.student_ema, device)
        policy_D = StudentPolicy(dp_d, diff_d, cfg_d, device)
        results["D_student_25step_ema"] = run_eval(
            "D. Student 25-step — EMA weights",
            policy_D, env, args.n_rollouts, horizon
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Label':<30s}  {'SR':>6s}  {'Num':>5s}")
    print("-" * 45)
    labels = {
        "A_teacher_50step":    "A. Teacher 50-step DDIM",
        "B_teacher_25step":    "B. Teacher 25-step DDIM",
        "C_student_25step_raw":"C. Student 25-step raw",
        "D_student_25step_ema":"D. Student 25-step EMA",
    }
    for key, label in labels.items():
        if key in results:
            r = results[key]
            print(f"  {label:<28s}  {r['Success_Rate']:>6.2f}  {int(r['Num_Success']):>5d}/{args.n_rollouts}")
    print()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True,
                   help="original robomimic .pth (for A & B)")
    p.add_argument("--student_ema",  default=None,
                   help="distilled student.pt with EMA weights (for D)")
    p.add_argument("--student_raw",  default=None,
                   help="distilled student_raw.pt with raw weights (for C)")
    p.add_argument("--n_rollouts",   type=int, default=20)
    p.add_argument("--horizon",      type=int, default=None)
    p.add_argument("--seed",         type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
