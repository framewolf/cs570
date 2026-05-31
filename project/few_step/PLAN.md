# Few-Step Diffusion Policy — Plan

## Goal
Apply fast diffusion samplers at **evaluation time** to a normally-trained
Diffusion Policy, and compare **success rate** vs **evaluation wall-clock time**
across samplers. Training is plain DDPM and is **never modified**.

## Constraints (from user)
1. **Do not touch the training code** (robomimic `diffusion_policy.py` etc.).
2. After the current `can` full training (2000 epochs) finishes, **apply immediately**.
3. Evaluate the same checkpoint under three samplers:
   - **DDPM** (default diffusion, 100 steps) — baseline
   - **DDIM** (10 steps)
   - **UniPC** (10 steps) — vendored implementation
4. Produce a **table** of Success Rate **and** evaluation completion time per sampler.

## Why this works without touching training
The DP inference loop is sampler-agnostic:
```python
self.noise_scheduler.set_timesteps(N)
for k in self.noise_scheduler.timesteps:
    eps = noise_pred_net(sample, k, global_cond)
    sample = self.noise_scheduler.step(eps, k, sample).prev_sample
```
So we just swap `noise_scheduler` and the step count at eval time. UniPC is
**training-free**, so the DDPM-trained checkpoint is reused as-is.

## Module layout (all new, training code untouched)
```
project/few_step/
├── unipc_scheduler.py   # vendored diffusers-0.16.0 UniPC, adapted to run on the
│                        #   diffusers-0.11.1 base classes pinned by robomimic.
│                        #   changes: imports, _compatibles=[], einsum ellipsis
│                        #   (shape-agnostic for [B,T,A] actions), + clip_sample.
├── samplers.py          # make_scheduler("ddpm"|"ddim"|"unipc", base, clip_sample)
├── patch.py             # apply_sampler() swaps sampler; restore_ema() loads the
│                        #   saved EMA weights (deserialize() doesn't) — required.
└── __init__.py
project/scripts/eval_fewstep.py   # loads 1 checkpoint, sweeps samplers, prints table
project/scripts/eval_fewstep.sh   # convenience wrapper (env isolation + paths)
```

## Key correctness points (validated)
- **Noise schedule must match training**: num_train_timesteps=100,
  beta_schedule="squaredcos_cap_v2", prediction_type="epsilon". (read from ckpt config)
- **EMA restore**: robomimic saves `model["ema"]` but only loads `model["nets"]`;
  inference uses the EMA model, so we inject the saved EMA weights. Without this,
  success rate collapses to ~0.
- **clip_sample**: actions ∈ [-1,1]; UniPC gained a clamp to match DDPM/DDIM.
- **Fair comparison**: seed reset before each sampler → identical initial states
  and identical starting noise.

## Validation (epoch-1200 checkpoint, 5 rollouts each) — PASSED
| Sampler | NFE | Success | Total time | Speedup |
|---------|-----|---------|-----------|---------|
| DDPM    | 100 | 100%    | 49.1 s    | 1.00x   |
| DDIM    | 10  | 100%    | 10.1 s    | 4.87x   |
| UniPC   | 10  | 100%    | 12.6 s    | 3.89x   |

## Final run (after training completes)
```bash
bash scripts/eval_fewstep.sh <final_checkpoint.pth>
# = eval_fewstep.py --n_rollouts 50 --horizon 400 --seed 0 \
#       --samplers ddpm:100 ddim:10 unipc:10
```
Deliver the final Success-Rate / time table (50 rollouts for statistical signal).
Optionally add `unipc:5` to probe the extreme few-step regime.
