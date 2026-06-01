#!/usr/bin/env python3
import csv
import json
import os
import sys


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python3 summarize_consistency_results.py <json_dir> <summary_csv> <summary_json>",
            file=sys.stderr,
        )
        return 1

    json_dir, csv_path, json_path = sys.argv[1:]
    rows = []

    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json"):
            continue
        task = os.path.splitext(name)[0]
        path = os.path.join(json_dir, name)
        with open(path, "r") as f:
            by_step = json.load(f)

        for step, stats in sorted(by_step.items(), key=lambda item: int(item[0])):
            row = {
                "task": task,
                "consistency_steps": int(step),
                "Success_Rate": stats.get("Success_Rate"),
                "Return": stats.get("Return"),
                "Horizon": stats.get("Horizon"),
                "Num_Success": stats.get("Num_Success"),
                "success_time_sec": stats.get(
                    "success_time_sec", stats.get("Success_Time_Sec")
                ),
                "n_rollouts": stats.get("n_rollouts"),
                "horizon": stats.get("horizon"),
            }
            rows.append(row)

    fieldnames = [
        "task",
        "consistency_steps",
        "Success_Rate",
        "Return",
        "Horizon",
        "Num_Success",
        "success_time_sec",
        "n_rollouts",
        "horizon",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)

    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
