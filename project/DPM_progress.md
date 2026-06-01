# DPM-Solver++ Evaluation Progress

**Date**: 2026-06-01  
**Branch**: DPM-solver/changed  
**Status**: In progress (9/15 done)

## Key Fix
Training used `DDPMScheduler(clip_sample=True)`.  
`DPMSolverMultistepScheduler` does not support `clip_sample`, so we patch  
`convert_model_output` to clamp predicted x_0 to [-1, 1].

## Completed Results (50 rollouts each)

### LIFT (baseline DDPM SR=1.00)
| Steps | Success Rate | Num_Success | Horizon |
|-------|-------------|-------------|---------|
| 6     | 1.00        | 50/50       | 43.7    |
| 3     | 1.00        | 50/50       | 43.8    |
| 1     | 0.08        | 4/50        | 374.7   |

### CAN (baseline DDPM SR=0.98)
| Steps | Success Rate | Num_Success | Horizon |
|-------|-------------|-------------|---------|
| 6     | 1.00        | 50/50       | 111.7   |
| 3     | 0.98        | 49/50       | 119.8   |
| 1     | 0.00        | 0/50        | 400.0   |

### SQUARE (baseline DDPM SR=0.86)
| Steps | Success Rate | Num_Success | Horizon |
|-------|-------------|-------------|---------|
| 6     | 0.80        | 40/50       | 198.1   |
| 3     | 0.76        | 38/50       | 210.7   |
| 1     | 0.00        | 0/50        | 400.0   |

## Pending
- transport × 6/3/1 (running...)
- tool_hang × 6/3/1 (queued)
