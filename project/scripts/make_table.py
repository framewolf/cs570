"""Quick table generator from eval_unipc JSON files.

Reads project/outputs/eval_unipc/*_unipc_*step.json and prints a markdown table
with the same shape as the deis branch's make_table.py.

Usage:
    python project/scripts/make_table.py [eval_dir]
    python project/scripts/make_table.py --markdown out.md
"""
import argparse
import glob
import json
import os
import sys


TASK_ORDER = ["lift", "can", "square", "transport", "tool_hang"]
STEP_ORDER = [6, 3, 1]
TASK_LABEL = {
    "lift": "Lift", "can": "Can", "square": "Square",
    "transport": "Transport", "tool_hang": "Tool Hang",
}


def render(results, fout):
    p = lambda s: print(s, file=fout)

    p("\n### Task × Step Success Rate\n")
    header = "| Task      | " + " | ".join(f"{s}-step       " for s in STEP_ORDER) + " |"
    sep = "|-----------|" + "|".join("---------------" for _ in STEP_ORDER) + "|"
    p(header)
    p(sep)
    for t in TASK_ORDER:
        row = f"| {TASK_LABEL[t]:9s} |"
        for s in STEP_ORDER:
            if (t, s) in results:
                d = results[(t, s)]
                row += f" {d['Success_Rate']*100:5.1f}% ({d['Num_Success']:2d}/{d['N_Rollouts']}) |"
            else:
                row += " (진행 중)     |"
        p(row)

    p("\n### Detailed Metrics\n")
    p("| Task | Step | Success Rate | Num Success | Horizon | Success Time (s) | Latency (ms/step) | Eval Wall Time (s) |")
    p("|------|------|--------------|-------------|---------|------------------|-------------------|--------------------|")
    for t in TASK_ORDER:
        for s in STEP_ORDER:
            if (t, s) in results:
                d = results[(t, s)]
                p(f"| {TASK_LABEL[t]} | {s}-step | {d['Success_Rate']*100:.1f}% | "
                  f"{d['Num_Success']}/{d['N_Rollouts']} | {d['Horizon']:.1f} | "
                  f"{d['Success_Time_s']:.2f} | {d['Avg_Latency_ms_per_step']:.1f} | "
                  f"{d['Eval_Wall_Time_s']:.1f} |")
            else:
                p(f"| {TASK_LABEL[t]} | {s}-step | (진행 중) | - | - | - | - | - |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", default=None,
                    help="dir containing *_unipc_*step.json; defaults to "
                         "$CS570_ROOT/project/outputs/eval_unipc")
    ap.add_argument("--markdown", default=None, help="also write the table to this file")
    args = ap.parse_args()

    eval_dir = args.eval_dir or os.path.join(
        os.environ.get("CS570_ROOT", "."), "project", "outputs", "eval_unipc"
    )

    results = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "*_unipc_*step.json"))):
        d = json.load(open(f))
        results[(d["Task"], d["Steps"])] = d

    render(results, sys.stdout)
    print(f"\n결과 폴더: {eval_dir}")
    expected = len(TASK_ORDER) * len(STEP_ORDER)
    print(f"완료된 평가: {len(results)} / {expected}")

    if args.markdown:
        with open(args.markdown, "w") as fout:
            render(results, fout)
            print(f"\nresults dir: {eval_dir}", file=fout)
            print(f"completed evals: {len(results)} / {expected}", file=fout)
        print(f"\nwrote markdown: {args.markdown}")


if __name__ == "__main__":
    main()
