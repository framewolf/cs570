# Progressive Distillation for Few-Step Diffusion Policy

Implementation of **Progressive Distillation** (Salimans & Ho, ICLR 2022) applied to
robomimic Diffusion Policy. Distills a 100-step DDPM teacher into 1-step student models
across 5 robosuite tasks.

---

## Method Overview

Each distillation round halves the number of inference steps:

```
Teacher (100 step) → 50 → 25 → 12 → 6 → 3 → 1  (6 rounds)
```

Per round:
1. Student weights initialized from teacher EMA
2. For each batch: teacher runs **2 DDPM steps** from x_t → target; student runs **1 step**
3. Loss: MSE in **x0-prediction space** (v-prediction parameterization)
4. Student EMA updated every iteration

### v-prediction parameterization

The student UNet predicts **v** (not ε), where:

```
v = (ε_pred − σ·z) / α
```

`α = sqrt(alphas_cumprod[t])`, `σ = sqrt(1 − alphas_cumprod[t])`, `z` = noisy sample.

Inference uses `GaussianDiffusionDefault.p_sample_loop` from the diffusion_distiller framework.
**Do not use standard DDPM/DDIM inference** — the distilled UNet no longer predicts ε.

---

## New Files

### `project/scripts/`

| File | Description |
|------|-------------|
| `dp_module.py` | `DPNetwork` (ConditionalUnet1D + eps→v wrapper), `RobomimicHDF5Dataset`, `make_condition()` |
| `convert_ckpt.py` | Convert robomimic `.pth` → distiller format `{G, n_timesteps, time_scale, ...}` |
| `v_to_eps_wrapper.py` | Load `student.pt` → `DPNetwork` + `GaussianDiffusionDefault` for inference |
| `run_distill.py` | One distillation round; saves EMA + raw checkpoints; wandb logging |
| `distill_all.sh` | Full 100→50→…→1 pipeline for a single task |
| `distill_one.sh` | Single distillation level helper |
| `run_all_tasks.sh` | Distill + eval all 5 tasks sequentially |
| `eval_student.py` | Correct v-prediction rollout evaluation |
| `eval_ablation.py` | A/B/C/D ablation (teacher DDIM vs student raw vs student EMA) |
| `record_videos.sh` | Record `rollout.mp4` per checkpoint folder |

### `project/configs/`

| File | Description |
|------|-------------|
| `dp_lowdim_pd.json` | Hyperparameter reference config for the PD pipeline (not a robomimic algo config) |

---

## Modified Files

### `robomimic/robomimic/utils/torch_utils.py`

Added a guard in `lr_scheduler_from_optim_params()` to handle inference-time checkpoint
loading where `num_train_batches` / `num_epochs` are not injected by `train.py`:

```python
if not isinstance(num_train_batches, int) or not isinstance(num_epochs, int):
    return None
```

---

## Checkpoint Format

```
project/distill_ckpts/{task}/
  teacher_step100/
    teacher.pt              ← converted from robomimic .pth
  student_step050/
    student.pt              ← EMA weights (use for eval)
    student_raw.pt          ← raw weights (ablation)
    rollout.mp4             ← 3-rollout demo video
    iter_0005000/student.pt ← intermediate checkpoint every 5k iters
    iter_0010000/student.pt
    ...
  student_step025/
  student_step012/
  student_step006/
  student_step003/
  student_step001/
```

`student.pt` schema:
```python
{
    "G":                  state_dict,       # DPNetwork weights (EMA)
    "n_timesteps":        int,              # inference steps for this level
    "time_scale":         float,            # 1.0 * 2^level
    "robomimic_config":   str,              # JSON config from teacher
    "shape_metadata":     dict,             # obs keys, action dim
    "env_metadata":       dict,             # for environment reconstruction
}
```

---

## Usage

### Full pipeline (all tasks)

```bash
cd $CS570_ROOT/project
bash scripts/run_all_tasks.sh 0 50000   # gpu=0, 50k iters per round
```

### Single task

```bash
bash scripts/distill_all.sh lift 0 50000 64 5000 diffusion-policy-pd
# args: task  gpu  num_iters  batch_size  save_every  wandb_project
```

### Evaluate a student checkpoint

```bash
python scripts/eval_student.py \
    --student_pt  distill_ckpts/lift/student_step006/student.pt \
    --orig_ckpt   ckpt_bundle/lift/last.pth \
    --n_rollouts  50 \
    --horizon     400 \
    --video_path  outputs/lift_step006.mp4
```

### A/B/C/D ablation

```bash
python scripts/eval_ablation.py \
    --teacher_ckpt ckpt_bundle/lift/last.pth \
    --student_ema  distill_ckpts/lift/student_step025/student.pt \
    --student_raw  distill_ckpts/lift/student_step025/student_raw.pt \
    --n_rollouts 20
```

---

## External Dependency

The pipeline uses the [diffusion_distiller](https://github.com/Hramchenko/diffusion_distiller)
library (lives at `$CS570_ROOT/diffusion_distiller/`, gitignored as external dep).
Our bridge code (`dp_module.py`, `convert_ckpt.py`, `v_to_eps_wrapper.py`) lives in
`project/scripts/` and is tracked in this repo.

Install (if not already):
```bash
pip install einops
# diffusion_distiller has no pip package — clone alongside project/
git clone https://github.com/Hramchenko/diffusion_distiller $CS570_ROOT/diffusion_distiller
```

---

## Results

**Setup:** 50k iterations per round, batch size 64, lr 3e-5, 50 rollouts evaluation.

| steps | lift | can | square | transport | tool_hang |
|-------|------|-----|--------|-----------|-----------|
| teacher (100) | — | — | — | — | — |
| **050** | **100%** | **100%** | 76% | 44% | 74% |
| **025** | 98% | **100%** | **78%** | **46%** | 56% |
| **012** | 90% | 98% | 74% | 38% | 26% |
| **006** | 50% | 64% | 32% | 2% | 0% |
| **003** | 16% | 0% | 0% | 0% | 0% |
| **001** | 2% | 0% | 0% | 0% | 0% |

### Key observations

- **lift / can**: step012 이상에서 90%+ 유지 — distillation 품질 우수
- **square / transport / tool_hang**: teacher 자체 성능이 낮아 distillation 상한 제한
- **step006 임계점**: 대부분 task에서 급격한 성능 하락
- **step001**: 1-step은 현재 방법으로 한계 (lift 2%, 나머지 0%)
- **step025 vs step050**: square/transport에서 025 > 050 — 누적 distillation 오차 영향
