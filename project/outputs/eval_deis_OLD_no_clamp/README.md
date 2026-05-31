# DEIS Sampler — 초기 시도 (BUG: action clipping 누락)

> **⚠️ 주의**: 이 폴더의 결과는 **DEIS scheduler에서 action clipping이 빠진 버그가 있던 상태**의 평가 결과입니다.
> 정정된 최종 결과는 `project/outputs/eval_deis/` 폴더의 README를 참조하세요.

## 버그 요약

DDPM/DDIM은 `clip_sample=True`로 학습되어 매 denoising step마다 sample을 `[-1, 1]`로 자른다.
그런데 `diffusers.DEISMultistepScheduler`는 `clip_sample` 인자 자체가 없고,
대안인 `thresholding`은 4D 이미지(batch, channels, H, W)만 지원해서 1D action sequence에는 적용 불가능하다.

따라서 DEIS 추론 시 sample이 `[-1, 1]`을 벗어나도 자르지 않은 채 누적되어,
특히 어려운 task(Transport, Tool Hang)에서 action이 완전히 무너지는 현상이 발생했다.

진단 과정에서 transport 20-step에 수동 clamp 추가만으로 0% → 40%(5 rollouts)로 회복되어
버그를 확정했고, 추론 루프에 `naction.clamp(-1.0, 1.0)`을 추가하여 정정했다.

## 결과 (50 rollouts each)

### Task × Step Success Rate

| Task      | 20-step       | 6-step      | 3-step      | 1-step      |
|-----------|---------------|-------------|-------------|-------------|
| Lift      | 100.0% (50/50)| 14.0% (7/50)| 10.0% (5/50)| 8.0% (4/50) |
| Can       | 72.0% (36/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| Square    | 32.0% (16/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| Transport | 0.0% (0/50)   | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| Tool Hang | 0.0% (0/50)   | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |

### Detailed Metrics

| Task | Step | Success Rate | Num Success | Horizon | Success Time (s) |
|------|------|--------------|-------------|---------|------------------|
| Lift | 20-step | 100.0% | 50/50 | 43.4 | 2.17 |
| Lift | 6-step  | 14.0% | 7/50 | 365.4 | 18.27 |
| Lift | 3-step  | 10.0% | 5/50 | 375.5 | 18.77 |
| Lift | 1-step  | 8.0% | 4/50 | 383.0 | 19.15 |
| Can  | 20-step | 72.0% | 36/50 | 200.1 | 10.01 |
| Can  | 6-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Can  | 3-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Can  | 1-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Square | 20-step | 32.0% | 16/50 | 324.0 | 16.20 |
| Square | 6-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Square | 3-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Square | 1-step  | 0.0% | 0/50 | 400.0 | 20.00 |
| Transport | 20-step | 0.0% | 0/50 | 700.0 | 35.00 |
| Transport | 6-step  | 0.0% | 0/50 | 700.0 | 35.00 |
| Transport | 3-step  | 0.0% | 0/50 | 700.0 | 35.00 |
| Transport | 1-step  | 0.0% | 0/50 | 700.0 | 35.00 |
| Tool Hang | 20-step | 0.0% | 0/50 | 700.0 | 35.00 |
| Tool Hang | 6-step  | 0.0% | 0/50 | 700.0 | 35.00 |
| Tool Hang | 3-step  | 0.0% | 0/50 | 700.0 | 35.00 |
| Tool Hang | 1-step  | 0.0% | 0/50 | 700.0 | 35.00 |

## 후속 조치

추론 루프(`robomimic/algo/diffusion_policy.py`)에 DEIS-only clamp 추가:

```python
naction = self.noise_scheduler.step(...).prev_sample
if self.algo_config.deis.enabled:
    naction = naction.clamp(-1.0, 1.0)
```

진단(transport 20-step, 5 rollouts) 결과 0% → 40%(2/5)로 회복.
전체 재평가 결과는 `project/outputs/eval_deis/` 참조.
