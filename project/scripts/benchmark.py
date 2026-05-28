import json
import time
import numpy as np

latencies = []
successes = []
horizons = []

def benchmark(policy, env, num_rollouts=10, horizon=400):

    for rollout_idx in range(num_rollouts):

        obs = env.reset()
        done = False

        step_count = 0
        rollout_success = False

        while not done and step_count < horizon:

            start = time.perf_counter()

            action = policy(obs)

            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            obs, reward, done, info = env.step(action)

            step_count += 1

            if info.get("success", False):
                rollout_success = True
                break

        successes.append(float(rollout_success))
        horizons.append(step_count)

    results = {
        "success_rate": float(np.mean(successes)),
        "avg_horizon": float(np.mean(horizons)),
        "avg_success_time_sec": float(np.mean(horizons) / 20.0),
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }

    print(json.dumps(results, indent=2))

    return results
