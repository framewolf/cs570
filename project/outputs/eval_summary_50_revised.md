# DMD2 Eval Summary (50 Rollouts)

**Reference format:** `output_summary.md`의 `Results`, `결과`, `정리`, 모델 설명 구조를 참고하여 정리  
**Evaluation file:** `eval_summary_50.md`  
**Rollouts:** 50 rollouts per task / step setting  
**Control frequency:** 20 Hz 기준으로 `Approx Success Time (s) = Horizon / 20.0`

---

## Results

| Task | Steps | Success Rate | Num Success | Avg Horizon | Approx Success Time (s) |
|------|------:|-------------:|------------:|------------:|------------------------:|
| **Lift** | 1 | 0.06 | 3/50 | 385.6 | 19.278 |
| | 3 | 1.00 | 50/50 | 43.8 | 2.189 |
| | 6 | 1.00 | 50/50 | 43.8 | 2.192 |
| **Can** | 1 | 0.00 | 0/50 | 400.0 | 20.000 |
| | 3 | 1.00 | 50/50 | 112.9 | 5.643 |
| | 6 | 0.96 | 48/50 | 122.1 | 6.107 |
| **Square** | 1 | 0.00 | 0/50 | 400.0 | 20.000 |
| | 3 | 0.76 | 38/50 | 209.5 | 10.473 |
| | 6 | 0.76 | 38/50 | 210.3 | 10.513 |
| **Transport** | 1 | 0.00 | 0/50 | 700.0 | 35.000 |
| | 3 | 0.16 | 8/50 | 668.4 | 33.421 |
| | 6 | 0.24 | 12/50 | 647.7 | 32.385 |
| **Tool Hang** | 1 | 0.00 | 0/50 | 700.0 | 35.000 |
| | 3 | 0.24 | 12/50 | 658.9 | 32.945 |
| | 6 | 0.14 | 7/50 | 672.7 | 33.635 |

---

## 결과

- `Approx Success Time (s) = Horizon / 20.0`이며, Horizon과 동일한 정보를 시간 단위로 환산한 값이다.
- Lift는 3-step과 6-step 모두 성공률 100%를 달성했다. 1-step에서도 6% 성공이 관찰되었지만, 평균 horizon이 385.6으로 매우 길기 때문에 안정적인 policy라고 보기는 어렵다.
- Can은 3-step에서 성공률 100%를 달성했지만, 6-step에서는 96%로 소폭 하락했다. 즉, Can task에서는 추가 refinement step이 반드시 성능 향상으로 이어지지는 않았다.
- Square는 3-step과 6-step 모두 76%로 동일했다. 이 task에서는 step 수 증가가 명확한 성능 개선을 만들지 않았으며, 현재 student의 trajectory distribution 또는 task-specific alignment 품질이 성능 병목일 가능성이 있다.
- Transport는 3-step 16%에서 6-step 24%로 상승했다. 성공률 자체는 아직 낮지만, long-horizon transport task에서는 추가 denoising/refinement가 일부 도움이 되는 경향을 보였다.
- Tool Hang은 3-step 24%에서 6-step 14%로 하락했다. 이는 step 수가 많아지면서 오히려 re-noising / denoising 과정에서 필요한 정밀 삽입 또는 걸기 동작이 흐트러질 수 있음을 시사한다.
- 1-step은 Lift를 제외한 모든 task에서 성공률 0%였다. 따라서 현재 DMD2 student는 high-noise action trajectory를 한 번에 실행 가능한 robot action sequence로 복원하기에는 아직 충분하지 않다.
- 전체적으로 3-step은 단순한 task에서 충분히 강하고, 6-step은 Transport처럼 긴 horizon이 필요한 task에서 일부 이득을 보이지만 모든 task에 대해 일관되게 우월하지는 않다.

---

## 정리

이번 50-rollout 평가에서 DMD2 기반 student policy는 task complexity에 따라 서로 다른 step-scaling 양상을 보였다. Lift와 Can처럼 비교적 짧은 horizon의 grasp-and-lift 또는 pick-and-place 계열 task에서는 3-step inference만으로도 거의 완전한 성능에 도달했다. Lift는 3-step과 6-step 모두 100% 성공률을 보였고, Can은 3-step에서 100%, 6-step에서 96%를 기록했다. 따라서 이 두 task에서는 6-step으로 늘리는 것이 실질적인 이득을 주지 않으며, 오히려 Can에서는 아주 작은 성능 하락이 나타났다.

Square는 3-step과 6-step 모두 76%로 동일했다. 이는 Square task가 단순히 denoising step 수를 늘린다고 해결되는 문제가 아니라, 정밀한 위치 정렬, 접촉 안정성, 또는 task-specific action distribution의 학습 품질에 더 강하게 의존할 수 있음을 의미한다. 즉, 추가 sampling step보다 student가 학습한 action distribution 자체의 품질이나 reward/task supervision이 더 중요한 병목일 가능성이 있다.

Transport는 3-step 16%에서 6-step 24%로 상승했다. Transport는 긴 horizon 동안 물체를 안정적으로 유지하면서 이동해야 하므로, action sequence 후반부의 누적 오차가 성공률에 큰 영향을 준다. 이 경우 6-step refinement는 3-step보다 더 안정적인 action trajectory를 제공한 것으로 볼 수 있다. 다만 절대 성공률은 여전히 낮기 때문에, DMD2 distillation 자체만으로 충분하다기보다는 closed-loop correction, task-conditioned refinement, 또는 stronger teacher trajectory가 추가로 필요해 보인다.

반면 Tool Hang은 3-step 24%에서 6-step 14%로 하락했다. Tool Hang은 물체를 단순히 운반하는 것뿐 아니라, 최종 위치에서 정밀하게 걸거나 삽입하는 alignment가 중요하다. 이 task에서는 반복적인 re-noising과 denoising이 trajectory를 더 좋게 refinement하기보다는, 이미 만들어진 정밀한 action을 흐트러뜨리거나 error accumulation을 만들었을 가능성이 있다. 따라서 Tool Hang에서는 step 수를 늘리는 것보다 stop timestep schedule, noise schedule, 또는 final-stage precision loss를 조정하는 것이 더 중요할 수 있다.

종합하면, 현재 DMD2 student policy는 1-step policy로 바로 사용하기에는 부족하지만, 3-step 또는 6-step inference를 사용하면 일부 robot manipulation task에서 의미 있는 성능을 낼 수 있다. 단순한 task에서는 3-step이 충분하고, Transport처럼 long-horizon 특성이 강한 task에서는 6-step이 도움이 될 수 있다. 그러나 Tool Hang처럼 final precision이 중요한 contact-rich task에서는 step 수 증가가 오히려 성능을 떨어뜨릴 수 있으므로, task별 inference step 수와 denoising schedule을 따로 선택하는 것이 필요하다.

---

## Distribution Matching Distillation 관점에서의 해석

DMD2의 핵심은 teacher diffusion policy의 sampling trajectory를 그대로 1:1 regression하는 것이 아니라, student generator가 만들어내는 action trajectory distribution이 teacher 또는 expert trajectory distribution과 가까워지도록 학습하는 것이다. 즉, 목표는 특정 noisy action input에 대해 teacher가 만든 exact action sequence를 복사하는 것이 아니라, 전체적으로 성공 가능한 action trajectory의 분포를 student가 빠르게 생성하도록 만드는 것이다.

로봇 조작 문제에서 diffusion sample은 이미지가 아니라 action trajectory이다. 따라서 DMD2를 로봇에 적용하면 다음과 같은 형태로 해석할 수 있다.

```text
image DMD2:
    noise z → student generator → image

robot DMD2:
    noise z + observation obs → student policy → action trajectory
```

Teacher DDPM policy는 여러 denoising step을 거쳐 action trajectory를 생성한다. 반면 DMD2 student는 훨씬 적은 step으로 action trajectory를 생성해야 한다. 이때 student가 teacher의 모든 denoising path를 정확히 따라 하도록 강제하면, teacher의 특정 sampling path에 묶이게 되고 student의 일반화나 fast inference 장점이 제한될 수 있다. Distribution Matching Distillation은 이 문제를 완화하기 위해 student output distribution 자체를 teacher / successful trajectory distribution과 맞추는 방향을 사용한다.

이를 action trajectory 관점에서 쓰면, teacher score와 fake score의 차이를 이용해 student를 업데이트하는 구조로 볼 수 있다.

```math
s_{teacher}(x_t, t, obs) - s_{fake}(x_t, t, obs)
```

여기서 `x_t`는 noisy action trajectory이고, `s_teacher`는 frozen teacher diffusion policy가 제공하는 score이며, `s_fake`는 현재 student가 생성하는 fake action trajectory distribution의 score이다. 직관적으로 `s_teacher`는 teacher distribution 쪽으로 가야 할 방향을 알려주고, `s_fake`는 현재 student distribution이 이미 차지하고 있는 방향을 보정한다. 두 score의 차이를 사용하면 student trajectory distribution을 teacher trajectory distribution 쪽으로 이동시킬 수 있다.

이번 결과에서 1-step 성능이 대부분 실패한 것은, student가 teacher distribution을 한 번의 mapping으로 충분히 복원하지 못하고 있음을 보여준다. 특히 action trajectory는 이미지보다 물리적 제약이 강하기 때문에, distribution matching만으로는 contact timing, object alignment, torque feasibility, long-horizon consistency를 모두 만족시키기 어렵다. 따라서 3-step 또는 6-step inference를 사용하면 student가 중간 refinement를 통해 더 teacher-like하고 physically plausible한 trajectory를 만들 수 있다.

그러나 step 수가 증가한다고 항상 성능이 좋아지지는 않았다. Can과 Tool Hang에서 6-step이 3-step보다 낮은 성능을 보인 것은, repeated denoising/refinement가 task-specific precision을 항상 보존하지는 않는다는 점을 보여준다. 특히 Tool Hang처럼 마지막 정렬이 중요한 task에서는 distribution-level matching이 평균적으로 그럴듯한 trajectory를 만들더라도, 성공을 결정하는 final contact geometry까지 충분히 보장하지 못할 수 있다. 따라서 DMD2-style policy를 로봇 조작에 적용할 때는 distribution matching objective에 더해 task success loss, final pose alignment loss, contact consistency loss, closed-loop rollout evaluation을 함께 고려해야 한다.

정리하면, DMD2의 distribution matching distillation은 robot diffusion policy를 빠르게 만드는 데 유용한 방향이지만, 로봇에서는 단순히 teacher distribution을 맞추는 것만으로 충분하지 않다. 성공률을 높이기 위해서는 task별 inference step 수 선택, backward simulation 기반 train-test mismatch 완화, 그리고 trajectory feasibility를 직접 반영하는 loss가 함께 필요하다.

---

## Detailed Metrics

| Task | Step | Success Rate | Num Success | Return | Horizon | Approx Success Time (s) |
|------|------|-------------:|------------:|-------:|--------:|------------------------:|
| Lift | 6-step | 100.0% | 50/50 | 1.000 | 43.8 | 2.192 |
| Lift | 3-step | 100.0% | 50/50 | 1.000 | 43.8 | 2.189 |
| Lift | 1-step | 6.0% | 3/50 | 0.060 | 385.6 | 19.278 |
| Can | 6-step | 96.0% | 48/50 | 0.960 | 122.1 | 6.107 |
| Can | 3-step | 100.0% | 50/50 | 1.000 | 112.9 | 5.643 |
| Can | 1-step | 0.0% | 0/50 | 0.000 | 400.0 | 20.000 |
| Square | 6-step | 76.0% | 38/50 | 0.760 | 210.3 | 10.513 |
| Square | 3-step | 76.0% | 38/50 | 0.760 | 209.5 | 10.473 |
| Square | 1-step | 0.0% | 0/50 | 0.000 | 400.0 | 20.000 |
| Transport | 6-step | 24.0% | 12/50 | 0.240 | 647.7 | 32.385 |
| Transport | 3-step | 16.0% | 8/50 | 0.160 | 668.4 | 33.421 |
| Transport | 1-step | 0.0% | 0/50 | 0.000 | 700.0 | 35.000 |
| Tool Hang | 6-step | 14.0% | 7/50 | 0.140 | 672.7 | 33.635 |
| Tool Hang | 3-step | 24.0% | 12/50 | 0.240 | 658.9 | 32.945 |
| Tool Hang | 1-step | 0.0% | 0/50 | 0.000 | 700.0 | 35.000 |

---
