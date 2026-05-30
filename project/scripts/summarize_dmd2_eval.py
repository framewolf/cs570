#!/usr/bin/env python3
import ast
import re
from pathlib import Path


def read_stats(log_path):
    text = log_path.read_text(errors="replace")
    match = re.search(r"Average Rollout Stats\s*\n(\{.*?\n\})", text, re.S)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def main():
    out_dir = Path("project/outputs/eval")
    summary_path = Path("project/outputs/eval_summary_50.md")
    steps = ["6", "3", "1"]
    tasks = ["lift", "can", "square", "transport", "tool_hang"]

    rows = []
    missing = []
    for step in steps:
        for task in tasks:
            log_path = out_dir / f"dmd2_{task}_{step}step_gan_eval.log"
            if not log_path.exists():
                missing.append(f"{log_path} (missing)")
                continue
            stats = read_stats(log_path)
            if stats is None:
                missing.append(f"{log_path} (incomplete)")
                continue
            horizon = float(stats["Horizon"])
            success_rate = float(stats["Success_Rate"])
            num_success = float(stats.get("Num_Success", success_rate * 50))
            rows.append(
                {
                    "task": task,
                    "step": f"{step}-step",
                    "success_rate": success_rate,
                    "num_success": num_success,
                    "return": float(stats["Return"]),
                    "horizon": horizon,
                    "approx_time": horizon / 20.0,
                }
            )

    lines = [
        "# DMD2 Eval Summary (50 Rollouts)",
        "",
        "| Task | Step | Success Rate | Num Success | Return | Horizon | Approx Success Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['step']} | {row['success_rate'] * 100:.1f}% | "
            f"{row['num_success']:.0f}/50 | {row['return']:.3f} | "
            f"{row['horizon']:.1f} | {row['approx_time']:.3f} |"
        )

    if missing:
        lines.extend(["", "Missing / incomplete logs:"])
        lines.extend(f"- {item}" for item in missing)

    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
