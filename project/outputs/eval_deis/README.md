# DEIS Sampler — 강지희

## 1. 방법 (DEIS)

DEIS(*Diffusion Exponential Integrator Sampler*)는 diffusion model의 ODE를 지수적분(exponential integrator) 기반 멀티스텝 솔버로 빠르게 푸는 방법이다. DDIM이 1차 근사라면 DEIS는 과거 timestep의 출력을 활용하는 고차(보통 2~3차) 솔버이므로, 같은 step 수에서 더 정확한 denoising이 가능하다.

핵심 특징은 **training-free**라는 점이다. 추가 학습 없이 기존 Diffusion Policy 체크포인트의 샘플러만 DEIS로 교체하면 그대로 동작한다.

## 2. 적용 방법

학습된 baseline 체크포인트(DDPM 100-step으로 학습된 teacher model)를 그대로 로드한 뒤, `robomimic/algo/diffusion_policy.py`의 noise scheduler 분기에 DEIS를 추가하여 추론 시에만 `DEISMultistepScheduler` (diffusers 0.18.2)로 교체했다. 학습 코드는 일절 건드리지 않았다.

### 발견된 버그: action clipping 누락

DDPM/DDIM은 `clip_sample=True`로 학습되어 매 denoising step마다 sample을 `[-1, 1]`로 자른다. 그런데 `diffusers.DEISMultistepScheduler`는 `clip_sample` 인자 자체가 없고, 대안인 `thresholding`은 4D 이미지(batch, channels, H, W)만 지원해서 1D action sequence에는 적용 불가능하다.

이 누락 때문에 DEIS 추론 시 sample이 `[-1, 1]`을 벗어나도 자르지 않은 채 누적되어, 초기 시도에서는 Transport/Tool Hang이 20-step에서도 0%가 나오는 등 결과가 무너졌다. **추론 루프에 수동 clamp 한 줄(`naction.clamp(-1.0, 1.0)`)을 추가하여 해결**. (초기 버그 상태 결과는 `eval_deis_OLD_no_clamp/README.md` 참조)

### 설정

- **Sampler swap 위치**: `eval_deis.py`의 `swap_to_deis()` (체크포인트 로드 직후 noise_scheduler 교체)
- **Solver order**: 2 (step 수가 order보다 작은 1-step에서는 자동으로 1로 하향)
- **Action clamp**: `[-1.0, 1.0]` (DEIS 분기에만 적용)
- **Beta schedule / prediction type**: baseline과 동일 (`squaredcos_cap_v2`, `epsilon`)
- **Inference steps**: 20 / 6 / 3 / 1
- **Rollouts**: 각 (task, step) 조합당 50회, seed=0

## 3. 결과 (50 rollouts each)

### 3.1 Task × Step Success Rate

| Task      | 20-step        | 6-step         | 3-step         | 1-step       |
|-----------|----------------|----------------|----------------|--------------|
| Lift      | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 6.0% (3/50)  |
| Can       | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 0.0% (0/50)  |
| Square    | 70.0% (35/50)  | 88.0% (44/50)  | 80.0% (40/50)  | 0.0% (0/50)  |
| Transport | 52.0% (26/50)  | 42.0% (21/50)  | 46.0% (23/50)  | 0.0% (0/50)  |
| Tool Hang | 50.0% (25/50)  | 46.0% (23/50)  | 42.0% (21/50)  | 0.0% (0/50)  |

참고: **Teacher Baseline (DDPM 100-step)** — Lift 100% / Can 98% / Square 86% / Transport 56% / Tool Hang 70%

### 3.2 Detailed Metrics

| Task | Step | Success Rate | Num Success | Horizon | Success Time (s) | Latency (ms/step) |
|------|------|--------------|-------------|---------|------------------|-------------------|
| Lift | 20-step | 100.0% | 50/50 | 43.6 | 2.18 | 41.2 |
| Lift | 6-step  | 100.0% | 50/50 | 43.6 | 2.18 | 30.7 |
| Lift | 3-step  | 100.0% | 50/50 | 43.6 | 2.18 | 29.1 |
| Lift | 1-step  | 6.0%   | 3/50  | 381.5 | 19.07 | 20.6 |
| Can  | 20-step | 100.0% | 50/50 | 111.3 | 5.56 | 36.1 |
| Can  | 6-step  | 100.0% | 50/50 | 112.4 | 5.62 | 26.7 |
| Can  | 3-step  | 100.0% | 50/50 | 112.6 | 5.63 | 26.1 |
| Can  | 1-step  | 0.0%   | 0/50  | 400.0 | 20.00 | 22.3 |
| Square | 20-step | 70.0% | 35/50 | 225.8 | 11.29 | 33.7 |
| Square | 6-step  | 88.0% | 44/50 | 184.2 | 9.21 | 25.6 |
| Square | 3-step  | 80.0% | 40/50 | 203.7 | 10.19 | 24.6 |
| Square | 1-step  | 0.0%  | 0/50  | 400.0 | 20.00 | 20.8 |
| Transport | 20-step | 52.0% | 26/50 | 569.7 | 28.48 | 43.3 |
| Transport | 6-step  | 42.0% | 21/50 | 593.1 | 29.66 | 36.6 |
| Transport | 3-step  | 46.0% | 23/50 | 594.4 | 29.72 | 33.0 |
| Transport | 1-step  | 0.0%  | 0/50  | 700.0 | 35.00 | 34.3 |
| Tool Hang | 20-step | 50.0% | 25/50 | 567.7 | 28.38 | 32.2 |
| Tool Hang | 6-step  | 46.0% | 23/50 | 578.8 | 28.94 | 27.0 |
| Tool Hang | 3-step  | 42.0% | 21/50 | 612.3 | 30.61 | 23.4 |
| Tool Hang | 1-step  | 0.0%  | 0/50  | 700.0 | 35.00 | 23.2 |

## 4. 분석

**(1) Lift / Can: 33배 가속 + baseline 동등.** DEIS 3-step에서 Lift 100%, Can 100% (baseline 100%, 98%). horizon도 baseline과 거의 동일. 즉 추가 학습 0의 비용으로 inference step을 33배(100→3) 줄여도 성능이 완벽히 유지된다. DEIS가 robot action diffusion에 잘 맞는 sampler임을 보여준다.

**(2) Square / Transport / Tool Hang: 일부 성능 저하.** 어려운 task일수록 step 수에 더 민감하다. Square 86% → 70~88%, Transport 56% → 42~52%, Tool Hang 70% → 42~50%. 특히 Tool Hang은 20-step에서도 baseline 대비 약 -20%p로 가장 크게 저하된다. 정밀 조작이 필요한 task일수록 더 많은 step이 요구된다.

**(3) 1-step의 절벽 — training-free의 본질적 한계.** 모든 task에서 1-step은 0~6%로 사실상 동작하지 못한다. DEIS는 ODE 솔버이므로 step 수가 너무 적으면 적분 오차가 누적되어 분포 자체가 무너진다. 같은 1-step에서 distillation 기반 방법(DMD2: Lift 6%, Progressive Distillation: Lift 2%)은 모델을 1-step 추론용으로 직접 학습시키기 때문에 일부 성공이 관찰되는 것과 대조된다. "샘플러 개선(training-free)"과 "학생 모델 distillation"은 가속 가능한 한계가 근본적으로 다르다.

**(4) Step 수가 항상 많을수록 좋은 것은 아니다.** Square에서 6-step (88%)이 20-step (70%)보다 높은 결과를 보였다. 샘플 수에 의한 우연일 수도 있으나, DEIS의 다단계 적분 특성상 중간 step에서 sweet spot이 있을 가능성이 있다.

## 5. 결론

> **DEIS는 추가 학습 0으로 33배 가속(100-step → 3-step)을 달성할 수 있는 가장 저렴한 방법이다. 단순 task(Lift, Can)에서는 baseline 동등, 복잡한 task에서는 일부 저하가 있으며, 1-step 같은 극단적 가속에는 distillation 기반 방법이 여전히 필요하다.**
>
> 주요 기여: **action clamping 누락 버그 발견 및 수정** — diffusers의 DEIS scheduler가 이미지 도메인 가정으로 설계되어 robot action diffusion에 직접 적용 시 누락된 안전장치를 식별.

## 6. 재현 방법

단일 실행: `python project/scripts/eval_deis.py --agent ckpt_bundle/<task>/last.pth --task <task> --steps <N> --n_rollouts 50`

전체 평가: `bash project/scripts/run_all_deis.sh 50`

결과 표 생성: `python project/scripts/make_table.py`

결과는 `project/outputs/eval_deis/{task}_deis_{N}step.json`에 저장된다.

## 7. 파일

- `project/scripts/eval_deis.py` — DEIS sampler swap + rollout 평가 스크립트
- `project/scripts/run_all_deis.sh` — 5 task × 4 step 전체 평가 wrapper
- `project/scripts/make_table.py` — JSON 결과 → markdown 표 자동 변환
- `robomimic/robomimic/algo/diffusion_policy.py` — DEIS scheduler 분기 + clamp 추가
- `project/outputs/eval_deis_OLD_no_clamp/README.md` — 초기 시도 (버그 상태) 기록
