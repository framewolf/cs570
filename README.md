# Few-Step Diffusion Policy on robomimic + robosuite

## Directory Layout

Assume the following structure. The base directory can be anywhere; the examples
below use `$HOME/cs570`.

```bash
export CS570_ROOT=${CS570_ROOT:-$HOME/cs570}

$CS570_ROOT/
├── robomimic/
├── robosuite/
└── project/
```

`project/` contains configs and scripts. `robomimic/` is tracked in this repo
because method implementations modify robomimic source files. `robosuite/` is
an external dependency and is cloned separately.

---

## 1. Environment Setup

```bash
conda create -n robomimic_dp python=3.10 -y
conda activate robomimic_dp
export CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
mkdir -p "$CS570_ROOT"
```

Install PyTorch(12.8기준). 쿠다 버전 맞춰서:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install robosuite:

```bash
cd "$CS570_ROOT"
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.5.1
pip install -r requirements.txt
pip install -e .
```

Install robomimic from the copy included in this repo:

```bash
cd "$CS570_ROOT"
cd robomimic
pip install -e .
```

Fix common dependencies:

```bash
pip install "numpy==1.24.4" "scipy==1.10.1" "pandas==1.5.3" "opencv-python==4.10.0.84"
```

Set environment variables:

```bash
export ROBOMIMIC_DIR=$CS570_ROOT/robomimic
export ROBOSUITE_DIR=$CS570_ROOT/robosuite
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

Optional: add them permanently.

```bash
cat >> ~/.bashrc <<'EOF'

export CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
export ROBOMIMIC_DIR=$CS570_ROOT/robomimic
export ROBOSUITE_DIR=$CS570_ROOT/robosuite
export MUJOCO_GL=egl
EOF

source ~/.bashrc
```

Create macro files:

```bash
python "$ROBOSUITE_DIR/robosuite/scripts/setup_macros.py"
python "$ROBOMIMIC_DIR/robomimic/scripts/setup_macros.py"
```

---

## 2. Check Environment

```bash
cd "$CS570_ROOT/project"
python scripts/check_env.py
```

Expected:

```bash
cuda available: True
robomimic import OK
robosuite import OK
```

---

## 3. Download Datasets

We use robomimic PH low_dim datasets.

Tasks:

```bash
lift
can
square
transport
tool_hang
```

Download:

```bash
cd "$CS570_ROOT/project"
bash scripts/download_lowdim_ph.sh
```

Check:

```bash
find $ROBOMIMIC_DIR/datasets -path "*/ph/*low_dim*.hdf5" | sort
```

Expected:

```bash
$ROBOMIMIC_DIR/datasets/can/ph/low_dim_v15.hdf5
$ROBOMIMIC_DIR/datasets/lift/ph/low_dim_v15.hdf5
$ROBOMIMIC_DIR/datasets/square/ph/low_dim_v15.hdf5
$ROBOMIMIC_DIR/datasets/tool_hang/ph/low_dim_v15.hdf5
$ROBOMIMIC_DIR/datasets/transport/ph/low_dim_v15.hdf5
```

---

## 4. Debug Simulation + Training

Run a debug test for all tasks. This uses robomimic `--debug`: 2 epochs, 3
training batches per epoch, and 2 rollout episodes of horizon 10 after each
epoch.

```bash
cd "$CS570_ROOT/project"
bash scripts/debug_all_tasks.sh
```

Success condition:

```bash
finished run successfully!
```

This checks:

```bash
dataset -> model init -> short training -> robosuite rollout -> video/offscreen rendering
```

---

## 5. Sanity Training

The sanity config trains for 100 epochs, with 100 training batches per epoch,
5 rollout episodes every 25 epochs, and rollout horizon 400.

Train Lift with the sanity config:

```bash
cd "$CS570_ROOT/project"
bash scripts/train_one.sh lift 0 configs/dp_lowdim_sanity.json
```

Train another task:

```bash
bash scripts/train_one.sh can 0 configs/dp_lowdim_sanity.json
bash scripts/train_one.sh square 0 configs/dp_lowdim_sanity.json
```

Arguments:

```bash
bash scripts/train_one.sh <task> <gpu_id> <config>
```

Example:

```bash
bash scripts/train_one.sh lift 0 configs/dp_lowdim_sanity.json
```

---

## 6. Full Training

```bash
cd "$CS570_ROOT/project"
bash scripts/train_one.sh lift 0 configs/dp_lowdim_full.json
```

---

## 7. Find Checkpoints

`train_one.sh` writes checkpoints under `project/results`.

```bash
cd "$CS570_ROOT/project"
find results -name "last.pth" | sort
```

Example checkpoint:

```bash
results/dp_lift_dp_lowdim_sanity/<timestamp>/last.pth
```

---

## 8. Evaluate Checkpoint

```bash
cd "$CS570_ROOT/project"

CKPT=$(find results -name "last.pth" | sort | tail -n 1)
bash scripts/eval_one.sh lift "$CKPT" 10 0
```

Arguments:

```bash
bash scripts/eval_one.sh <task> <checkpoint_path> <n_rollouts> <gpu_id>
```

Example:

```bash
bash scripts/eval_one.sh lift results/dp_lift_dp_lowdim_sanity/<timestamp>/last.pth 50 0
```

Outputs:

```bash
outputs/eval/lift_eval.log
outputs/eval/lift_eval.mp4
```

---

## 9. Read Evaluation Results

```bash
cat outputs/eval/lift_eval.log | tail -80
```

Important metrics:

```bash
Success_Rate
Return
Horizon
Num_Success
```

Approximate success time:

```bash
success_time_sec = Horizon / 20
```

because robosuite control frequency is 20 Hz.

Example:

```bash
Horizon = 43.1
success_time_sec = 43.1 / 20 = 2.155 sec
```

---

## 10. Notes

Default robomimic Diffusion Policy uses DDPM 100-step inference.

Built-in DDIM config exists in:

```bash
$ROBOMIMIC_DIR/robomimic/exps/templates/diffusion_policy.json
```

## 11. Team Workflow for robomimic Changes

This project changes robomimic source code, so robomimic is included directly in
the `cs570` repository. Do not let each teammate work only in independent
robomimic clones; make method work visible through branches and pull requests in
the shared `cs570` repo.

Recommended structure:

```bash
$CS570_ROOT/
├── project/      # shared experiment scripts, configs, README
├── robomimic/    # tracked source code for method changes
└── robosuite/    # ignored external dependency, checkout v1.5.1
```

For each method implementation:

```bash
cd "$CS570_ROOT"
git checkout main
git pull
git checkout -b feature/my-method
```

Edit files under `robomimic/` and `project/` as needed, then push the branch and
open a PR into `main`. This lets the team review code and experiment config
changes together.

The `cs570` repo should contain source code, configs, scripts, and README files.
Do not commit datasets, checkpoints, videos, logs, or result folders.
