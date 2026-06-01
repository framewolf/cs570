# DPM-Solver++ Evaluation Results

**Date**: 2026-06-01  
**Branch**: DPM-solver/changed  
**Status**: COMPLETE (15/15)

## Key Fix
Training used `DDPMScheduler(clip_sample=True)`.  
`DPMSolverMultistepScheduler` does not support `clip_sample`, so we patch  
`convert_model_output` to clamp predicted x_0 to [-1, 1].

## Results (50 rollouts each)

### LIFT (baseline DDPM SR=1.00)
| Steps | Success Rate | Num_Success | Horizon | Time(s) |
|-------|-------------|-------------|---------|---------|
| 6     | 1.00        | 50/50       | 43.7    | 2.19    |
| 3     | 1.00        | 50/50       | 43.8    | 2.19    |
| 1     | 0.08        | 4/50        | 374.7   | 18.73   |

### CAN (baseline DDPM SR=0.98)
| Steps | Success Rate | Num_Success | Horizon | Time(s) |
|-------|-------------|-------------|---------|---------|
| 6     | 1.00        | 50/50       | 111.7   | 5.59    |
| 3     | 0.98        | 49/50       | 119.8   | 5.99    |
| 1     | 0.00        | 0/50        | 400.0   | 20.00   |

### SQUARE (baseline DDPM SR=0.86)
| Steps | Success Rate | Num_Success | Horizon | Time(s) |
|-------|-------------|-------------|---------|---------|
| 6     | 0.80        | 40/50       | 198.1   | 9.91    |
| 3     | 0.76        | 38/50       | 210.7   | 10.53   |
| 1     | 0.00        | 0/50        | 400.0   | 20.00   |

### TRANSPORT (baseline DDPM: N/A)
| Steps | Success Rate | Num_Success | Horizon | Time(s) |
|-------|-------------|-------------|---------|---------|
| 6     | 0.42        | 21/50       | 598.9   | 29.94   |
| 3     | 0.36        | 18/50       | 622.9   | 31.15   |
| 1     | 0.00        | 0/50        | 700.0   | 35.00   |

### TOOL_HANG (baseline DDPM: N/A)
| Steps | Success Rate | Num_Success | Horizon | Time(s) |
|-------|-------------|-------------|---------|---------|
| 6     | 0.64        | 32/50       | 560.2   | 28.01   |
| 3     | 0.48        | 24/50       | 607.3   | 30.36   |
| 1     | 0.00        | 0/50        | 700.0   | 35.00   |
