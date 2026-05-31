"""Quick table generator from eval_deis JSON files."""
import json, glob, os, sys

eval_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("CS570_ROOT", "."), "project", "outputs", "eval_deis"
)

TASK_ORDER = ["lift", "can", "square", "transport", "tool_hang"]
STEP_ORDER = [20, 6, 3, 1]
TASK_LABEL = {"lift": "Lift", "can": "Can", "square": "Square",
              "transport": "Transport", "tool_hang": "Tool Hang"}

results = {}
for f in sorted(glob.glob(os.path.join(eval_dir, "*_deis_*step.json"))):
    d = json.load(open(f))
    results[(d["Task"], d["Steps"])] = d

print("\n### Task × Step Success Rate\n")
print("| Task      | 20-step       | 6-step        | 3-step        | 1-step        |")
print("|-----------|---------------|---------------|---------------|---------------|")
for t in TASK_ORDER:
    row = f"| {TASK_LABEL[t]:9s} |"
    for s in STEP_ORDER:
        if (t, s) in results:
            d = results[(t, s)]
            row += f" {d['Success_Rate']*100:5.1f}% ({d['Num_Success']:2d}/{d['N_Rollouts']}) |"
        else:
            row += " (진행 중)     |"
    print(row)

print("\n### Detailed Metrics\n")
print("| Task | Step | Success Rate | Num Success | Horizon | Success Time (s) | Latency (ms/step) |")
print("|------|------|--------------|-------------|---------|------------------|-------------------|")
for t in TASK_ORDER:
    for s in STEP_ORDER:
        if (t, s) in results:
            d = results[(t, s)]
            print(f"| {TASK_LABEL[t]} | {s}-step | {d['Success_Rate']*100:.1f}% | "
                  f"{d['Num_Success']}/{d['N_Rollouts']} | {d['Horizon']:.1f} | "
                  f"{d['Success_Time_s']:.2f} | {d['Avg_Latency_ms_per_step']:.1f} |")
        else:
            print(f"| {TASK_LABEL[t]} | {s}-step | (진행 중) | - | - | - | - |")
print(f"\n결과 폴더: {eval_dir}")
print(f"완료된 평가: {len(results)} / 20")
