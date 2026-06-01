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


def copy_ema_to_module(ema_state, module):
    shadow_params = ema_state.get("shadow_params", None)
    if shadow_params is None:
        raise RuntimeError("Unsupported EMA state format: missing shadow_params.")

    params = list(module.parameters())
    if len(params) != len(shadow_params):
        raise RuntimeError(
            "EMA parameter count mismatch: {} module params vs {} shadow params.".format(
                len(params), len(shadow_params)
            )
        )

    for param, shadow in zip(params, shadow_params):
        param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))


def eval_steps(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    base_ckpt = FileUtils.load_dict_from_checkpoint(args.agent)
    ema_state = base_ckpt["model"].get("ema", None)
    results = {}

    for num_steps in args.steps:
        ckpt = dict(base_ckpt)
        model_dict = dict(base_ckpt["model"])

        cfg = json.loads(ckpt["config"])
        cfg["algo"]["ddpm"]["enabled"] = False
        cfg["algo"]["ddim"]["enabled"] = True
        cfg["algo"]["ddim"]["num_train_timesteps"] = cfg["algo"]["ddpm"][
            "num_train_timesteps"
        ]
        cfg["algo"]["ddim"]["beta_schedule"] = cfg["algo"]["ddpm"]["beta_schedule"]
        cfg["algo"]["ddim"]["clip_sample"] = cfg["algo"]["ddpm"]["clip_sample"]
        cfg["algo"]["ddim"]["prediction_type"] = cfg["algo"]["ddpm"]["prediction_type"]
        cfg["algo"]["ddim"]["num_inference_timesteps"] = int(num_steps)

        # Avoid the local diffusers EMA constructor/load incompatibility during eval.
        cfg["algo"]["ema"]["enabled"] = False
        model_dict["ema"] = None

        ckpt["config"] = json.dumps(cfg)
        ckpt["model"] = model_dict

        print("")
        print("=" * 60)
        print("Evaluating DDIM num_inference_timesteps={}".format(num_steps))
        print("=" * 60)

        policy, loaded_ckpt = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt,
            device=device,
            verbose=(not args.quiet),
        )
        if not args.no_ema and ema_state is not None:
            copy_ema_to_module(ema_state, policy.policy.nets)
            policy.policy.set_eval()

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
            video_path = os.path.join(args.video_dir, "ddim_steps_{}.mp4".format(num_steps))
            video_writer = imageio.get_writer(video_path, fps=20)

        rollout_stats = []
        for rollout_i in range(args.n_rollouts):
            stats, _ = rollout(
                policy=policy,
                env=env,
                horizon=args.horizon,
                render=False,
                video_writer=video_writer if rollout_i < args.n_videos else None,
                video_skip=args.video_skip,
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
        results[str(num_steps)] = avg
        print("Average Rollout Stats for DDIM steps={}".format(num_steps))
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
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--camera_names", nargs="+", default=["agentview"])
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--n_videos", type=int, default=1)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    eval_steps(args)


if __name__ == "__main__":
    main()
