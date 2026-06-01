# Consistency Policy Eval Summary

**Run:** `eval_consistency_all/20260531_172448`
**Fix applied:** chaining timestep order ascending → descending (noise level 기준)

## Results

| Task | Steps | Success_Rate | Num_Success | Avg Horizon |
|------|-------|-------------|-------------|-------------|
| **lift** | 1 | 0.00 | 0 | 400.00 |
| | 3 | 1.00 | 50 | 43.68 |
| | 6 | 1.00 | 50 | 43.90 |
| **can** | 1 | 0.00 | 0 | 400.00 |
| | 3 | 1.00 | 50 | 110.78 |
| | 6 | 1.00 | 50 | 111.26 |
| **square** | 1 | 0.00 | 0 | 400.00 |
| | 3 | 0.76 | 38 | 205.80 |
| | 6 | 0.74 | 37 | 217.40 |
| **tool_hang** | 1 | 0.00 | 0 | 700.00 |
| | 3 | 0.36 | 18 | 626.08 |
| | 6 | 0.48 | 24 | 576.34 |
| **transport** | 1 | 0.00 | 0 | 700.00 |
| | 3 | 0.46 | 23 | 578.64 |
| | 6 | 0.60 | 30 | 551.66 |

## 결과

- `avg_success_time_sec = Horizon / 20.0` (고정 20Hz) — Horizon과 동일한 정보라 생략
- lift, can: steps=3에서 이미 SR=1.0 (ceiling), steps=6 비교 무의미
- tool_hang, transport: steps=6이 steps=3 대비 SR 각각 +0.12, +0.14 향상 → chaining fix 효과 확인
- *lift/can은 너무 쉬운 테스크라 3 step만으로 모두 성공시킴, 어려운 테스크인 Tool_hang, transport에서는 step수가 올라감에 따라 성능이 올라감을 보임 1->3->6*
- square: steps=6이 오히려 소폭 하락 (0.76 → 0.74), 학습 품질 이슈 가능성
- step=1은 모든 태스크에서 SR=0 (1-step consistency는 현재 모델로 불충분)

## 정리

로봇 조작 관점에서 chaining step의 효과는 task complexity에 따라 다르게 나타났다. Lift와 Can은 비교적 짧은 horizon의 grasp-and-lift / pick-and-place 계열 task로, 접촉 구간과 순차적 의사결정이 상대적으로 단순하다. 따라서 3-step consistency inference만으로도 안정적인 action trajectory를 생성할 수 있었고, 두 task 모두 성공률 100%에 도달했다.

반면 Tool Hang과 Transport는 더 긴 horizon과 여러 sub-stage를 요구하는 task이다. 물체를 잡는 것뿐 아니라 이동, 정렬, 삽입/걸기, 양팔 또는 장거리 운반에 가까운 복합 조작이 필요하므로 action sequence의 작은 오차가 후반부 실패로 이어지기 쉽다. 이러한 task에서는 6-step chaining이 3-step 대비 성공률을 각각 12%p, 14%p 향상시켰다. 이는 추가 denoising/refinement 단계가 복잡한 로봇 조작에서 더 정밀하고 안정적인 action trajectory를 만드는 데 도움이 되었음을 의미한다.

1-step inference가 모든 task에서 실패한 것은, 현재 student policy가 high-noise action trajectory를 한 번에 실행 가능한 robot action으로 복원하기에는 충분하지 않다는 것을 보여준다. 하지만 3-step 또는 6-step chaining을 사용하면 중간 refinement 과정을 통해 물리적으로 더 그럴듯한 action sequence를 생성할 수 있고, 특히 contact-rich하고 long-horizon인 task에서 효과가 커진다.

Square의 경우 6-step이 3-step보다 소폭 낮은 성공률을 보였는데, 이는 step 수가 많아진다고 항상 성능이 좋아지는 것은 아님을 보여준다. 반복적인 re-noising과 denoising이 오히려 정밀한 alignment 동작에 필요한 action을 흔들거나 error accumulation을 만들 수 있다. 따라서 consistency policy의 inference step 수는 task별로 최적점이 존재한다.

종합하면, Consistency Policy는 기존 DDPM의 많은 denoising step을 3~6 step으로 줄이면서도 의미 있는 로봇 조작 성능을 유지할 수 있었다. 단순한 manipulation task에서는 3-step이 충분하고, Tool Hang/Transport처럼 더 복잡한 long-horizon task에서는 6-step refinement가 더 안정적인 선택으로 나타났다.

## Consistency Model

Consistency Model은 diffusion trajectory 위의 임의의 noisy sample을 clean sample로 직접 매핑하는 consistency function을 학습하는 모델이다. 기존 DDPM이 각 timestep에서 noise ε를 예측하고 scheduler를 통해 점진적으로 sample을 복원하는 반면, Consistency Model은 특정 noise level의 sample을 더 낮은 noise level 또는 clean sample로 직접 매핑하는 함수를 학습한다.

로봇 Diffusion Policy에서는 sample이 이미지가 아니라 action trajectory이다. 따라서 Consistency Policy의 입력은 현재 observation, noisy action trajectory, 현재 timestep, 그리고 목표 stop timestep으로 구성된다. 모델은 이를 바탕으로 해당 noise level에서 목표 noise level의 action trajectory를 예측한다.

기존 DDPM Diffusion Policy는 다음과 같은 형태로 동작한다.

    εθ(obs, x_t, t) → predicted noise

즉, noisy action trajectory x_t에서 포함된 noise를 예측하고, scheduler가 이를 이용해 x_{t-1}을 계산한다. 반면 Consistency Policy는 다음과 같은 형태의 student policy를 학습한다.

    fθ(obs, x_t, t, s) → x_s

여기서 t는 현재 noise level, s는 목표 stop timestep이다. 즉, student는 noisy action trajectory를 받아 더 denoised된 action trajectory로 직접 이동시키는 mapping을 학습한다.

본 프로젝트에서는 미리 학습된 DDPM Diffusion Policy를 teacher로 사용하고, Consistency Policy student를 distillation 방식으로 학습한다. Teacher policy는 기존의 denoising trajectory를 제공하고, student는 teacher가 여러 denoising step에 걸쳐 만든 중간 또는 최종 action trajectory를 더 적은 step으로 근사하도록 학습된다.

구조적으로는 기존 Diffusion Policy의 Conditional 1D UNet backbone을 기반으로 하지만, 입력 조건과 학습 목표가 다르다. DDPM teacher는 timestep 하나를 condition으로 받아 noise를 예측하는 반면, Consistency Policy student는 현재 timestep과 stop timestep을 모두 condition으로 받아 denoised action trajectory를 직접 예측한다. 따라서 backbone은 유사하지만, 출력의 의미는 noise prediction이 아니라 action trajectory prediction에 가깝다.

Inference 시에는 Gaussian noise에서 시작한 action trajectory를 student가 바로 denoised action으로 변환한다. 또한 test-time chaining을 사용할 경우, 예측된 action에 다시 일정 noise level을 부여한 뒤 student를 반복적으로 적용하여 action trajectory를 refine한다. 이를 통해 기존 DDPM의 반복적인 denoising 과정을 훨씬 적은 network evaluation으로 대체한다.