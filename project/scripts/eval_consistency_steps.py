#!/usr/bin/env python
import argparse
import json
import os

import imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.run_trained_agent import rollout


def eval_steps(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    base_ckpt = FileUtils.load_dict_from_checkpoint(args.agent)
    results = {}

    for num_steps in args.steps:
        ckpt = dict(base_ckpt)
        cfg = json.loads(ckpt["config"])
        cfg["algo"]["consistency"]["inference"]["num_steps"] = int(num_steps)
        cfg["algo"]["consistency"]["inference"]["chaining_timesteps"] = []
        cfg["algo"]["consistency"]["teacher_checkpoint_path"] = None
        cfg["algo"]["consistency"]["warm_start"] = False
        ckpt["config"] = json.dumps(cfg)

        print("")
        print("=" * 60)
        print("Evaluating consistency inference num_steps={}".format(num_steps))
        print("=" * 60)

        policy, loaded_ckpt = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt,
            device=device,
            verbose=(not args.quiet),
        )
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=loaded_ckpt,
            env_name=args.env,
            render=False,
            render_offscreen=(args.video_dir is not None),
            verbose=(not args.quiet),
        )

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        video_writer = None
        if args.video_dir is not None:
            os.makedirs(args.video_dir, exist_ok=True)
            video_path = os.path.join(
                args.video_dir,
                "consistency_steps_{}.mp4".format(num_steps),
            )
            video_writer = imageio.get_writer(video_path, fps=20)

        rollout_stats = []
        for rollout_i in range(args.n_rollouts):
            stats, _ = rollout(
                policy=policy,
                env=env,
                horizon=args.horizon,
                render=False,
                video_writer=video_writer if rollout_i < args.n_videos else None,
                video_skip=5,
                return_obs=False,
                camera_names=args.camera_names,
            )
            rollout_stats.append(stats)

        if video_writer is not None:
            video_writer.close()
            print("Saved video to {}".format(video_path))

        flat = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        avg = {k: float(np.mean(flat[k])) for k in flat}
        avg["Num_Success"] = float(np.sum(flat["Success_Rate"]))
        avg["Success_Time_Sec"] = float(avg["Horizon"] / 20.0)
        avg["success_time_sec"] = avg["Success_Time_Sec"]
        avg["avg_success_time_sec"] = avg["Success_Time_Sec"]
        avg["n_rollouts"] = int(args.n_rollouts)
        avg["horizon"] = int(args.horizon)
        results[str(num_steps)] = avg
        print("Average Rollout Stats for num_steps={}".format(num_steps))
        print(json.dumps(avg, indent=4))

        del policy
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print("")
        print("Saved summary to {}".format(args.output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--camera_names", nargs="+", default=["agentview"])
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--n_videos", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    eval_steps(args)


if __name__ == "__main__":
    main()
