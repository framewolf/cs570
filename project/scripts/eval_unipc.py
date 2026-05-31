"""
UniPC sampler evaluation script.

Loads a trained Diffusion Policy checkpoint, replaces the sampler with UniPC at
a chosen number of inference steps, and runs n_rollouts in robosuite.
Output JSON layout mirrors eval_deis.py for shared table tooling.

Usage:
    python project/scripts/eval_unipc.py \
        --agent results/.../model_epoch_2000.pth --task lift --steps 6 --n_rollouts 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

# robomimic ships its own standalone-eval rollout in scripts/. Importing it keeps
# our per-episode protocol (reset + get_state + reset_to) identical to the
# reference eval harnessed by sibling branches (e.g. feature/deis).
import robomimic
sys.path.insert(0, os.path.join(robomimic.__path__[0], "scripts"))
from run_trained_agent import rollout

from few_step.patch import apply_sampler, load_ckpt_dict, restore_ema


TASK_HORIZON = {
    "lift": 400, "can": 400, "square": 400,
    "transport": 700, "tool_hang": 700,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, help="path to trained .pth checkpoint")
    p.add_argument("--task", required=True, choices=list(TASK_HORIZON.keys()))
    p.add_argument("--steps", type=int, required=True, help="UniPC inference steps (NFE)")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver_order", type=int, default=2)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # legacy (diffusers 0.11.1) checkpoint format is auto-migrated in-memory;
    # new-format checkpoints are loaded natively by deserialize().
    ckpt_dict = load_ckpt_dict(args.agent)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=False
    )
    restore_ema(policy, ckpt_dict)

    info = apply_sampler(policy, "unipc", args.steps, solver_order=args.solver_order)
    print(f"[unipc] sampler swapped: steps={args.steps}, solver_order={info['solver_order']}")

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
    )

    # identical initial states + starting noise across runs at the same seed
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
    avg["Sampler"] = "unipc"
    avg["Steps"] = args.steps
    avg["Solver_Order"] = info["solver_order"]
    avg["N_Rollouts"] = args.n_rollouts
    avg["Num_Success"] = int(np.sum(rs["Success_Rate"]))
    avg["Success_Time_s"] = avg["Horizon"] / 20.0   # robosuite control_freq = 20Hz
    avg["Avg_Latency_ms_per_step"] = float(np.mean(latencies))
    avg["Eval_Wall_Time_s"] = total_t

    out_dir = args.out_dir or os.path.join(
        os.environ.get("CS570_ROOT", "."), "project", "outputs", "eval_unipc"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.task}_unipc_{args.steps}step.json")
    with open(out_json, "w") as f:
        json.dump(avg, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(avg, indent=2))
    print(f"saved to: {out_json}")


if __name__ == "__main__":
    main()
