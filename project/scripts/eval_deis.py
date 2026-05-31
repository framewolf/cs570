"""
DEIS sampler evaluation script.

Loads a trained Diffusion Policy checkpoint, replaces the sampler with DEIS at
a chosen number of inference steps, and runs n_rollouts in robosuite.

Usage:
    python project/scripts/eval_deis.py \
        --agent ckpt_bundle/lift/last.pth --task lift --steps 6 --n_rollouts 50
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler

sys.path.insert(0, os.path.join(os.environ.get("ROBOMIMIC_DIR", ""), "robomimic", "scripts"))
from run_trained_agent import rollout


TASK_HORIZON = {
    "lift": 400, "can": 400, "square": 400,
    "transport": 700, "tool_hang": 700,
}


def swap_to_deis(policy, num_inference_steps, solver_order=2):
    algo = policy.policy
    cfg = algo.algo_config

    with cfg.values_unlocked():
        cfg.ddpm.enabled = False
        cfg.ddim.enabled = False
        cfg.deis.enabled = True
        cfg.deis.num_inference_timesteps = num_inference_steps
        cfg.deis.solver_order = solver_order

    new_scheduler = DEISMultistepScheduler(
        num_train_timesteps=cfg.deis.num_train_timesteps,
        beta_schedule=cfg.deis.beta_schedule,
        solver_order=cfg.deis.solver_order,
        prediction_type=cfg.deis.prediction_type,
        # thresholding=True,
        # sample_max_value=1.0,
    )
    algo.noise_scheduler = new_scheduler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_HORIZON.keys()))
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver_order", type=int, default=2)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )

    effective_order = min(args.solver_order, args.steps)
    swap_to_deis(policy, num_inference_steps=args.steps, solver_order=effective_order)
    print(f"[deis] sampler swapped: steps={args.steps}, solver_order={effective_order}")

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=True
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    horizon = TASK_HORIZON[args.task]
    stats_list = []
    latencies = []
    t_start = time.time()
    for i in range(args.n_rollouts):
        t0 = time.perf_counter()
        stats, _ = rollout(
            policy=policy, env=env, horizon=horizon,
            render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=["agentview"],
        )
        rollout_time = time.perf_counter() - t0
        latencies.append(rollout_time * 1000 / max(stats["Horizon"], 1))
        stats_list.append(stats)
        print(f"  [{i+1}/{args.n_rollouts}] success={stats['Success_Rate']} horizon={stats['Horizon']}")
    total_t = time.time() - t_start

    rs = TensorUtils.list_of_flat_dict_to_dict_of_list(stats_list)
    avg = {k: float(np.mean(rs[k])) for k in rs}
    avg["Task"] = args.task
    avg["Sampler"] = "deis"
    avg["Steps"] = args.steps
    avg["N_Rollouts"] = args.n_rollouts
    avg["Num_Success"] = int(np.sum(rs["Success_Rate"]))
    avg["Success_Time_s"] = avg["Horizon"] / 20.0
    avg["Avg_Latency_ms_per_step"] = float(np.mean(latencies))
    avg["Eval_Wall_Time_s"] = total_t

    out_dir = args.out_dir or os.path.join(
        os.environ.get("CS570_ROOT", "."), "project", "outputs", "eval_deis"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.task}_deis_{args.steps}step.json")
    with open(out_json, "w") as f:
        json.dump(avg, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(avg, indent=2))
    print(f"saved to: {out_json}")


if __name__ == "__main__":
    main()
