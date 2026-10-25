import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
EXPERIMENTS = ["pcb_yolox_s_e50", "pcb_yolox_s_neg_e50"]
SEEDS = range(5)
MAX_FA = 0.005
TARGET_RECALL = 0.995


def load_runs(exp, suffix):
    runs = []
    for seed in SEEDS:
        path = REPORTS / f"{exp}_seed{seed}_val_agnostic{suffix}.json"
        runs.append(json.loads(path.read_text()))
    return runs


def mean_std(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def best_recall(run, max_fa):
    ok = [row["board_recall"] for row in run["board_table"]
          if row["board_false_alarm"] <= max_fa]
    return max(ok) if ok else None


def summarize(exp, suffix):
    runs = load_runs(exp, suffix)
    thresholds = [row["threshold"] for row in runs[0]["board_table"]]
    for run in runs:
        assert [row["threshold"] for row in run["board_table"]] == thresholds

    summary = {"experiment": exp}
    for key in ["ap50", "ap50_95"]:
        summary[key] = mean_std([run[key] for run in runs])

    table = []
    for i, thr in enumerate(thresholds):
        recalls = [run["board_table"][i]["board_recall"] for run in runs]
        fas = [run["board_table"][i]["board_false_alarm"] for run in runs]
        table.append({"threshold": thr,
                      "recall": mean_std(recalls),
                      "false_alarm": mean_std(fas)})
    summary["board_table"] = table

    bests = [best_recall(run, MAX_FA) for run in runs]
    summary["best_recall_per_seed"] = bests
    summary["seeds_meeting_target"] = sum(
        b is not None and b >= TARGET_RECALL for b in bests)
    return summary


def print_summary(s):
    print(f"\n=== {s['experiment']} ===")
    print(f"AP50     {s['ap50'][0]:.3f} +- {s['ap50'][1]:.3f}")
    print(f"AP50:95  {s['ap50_95'][0]:.3f} +- {s['ap50_95'][1]:.3f}")
    print("thr    recall            false_alarm")
    for row in s["board_table"]:
        r, f = row["recall"], row["false_alarm"]
        print(f"{row['threshold']:<5}  {r[0]:.3f} +- {r[1]:.3f}   "
              f"{f[0]:.3f} +- {f[1]:.3f}")
    print(f"best recall with FA <= {MAX_FA}: {s['best_recall_per_seed']}")
    print(f"seeds meeting target: {s['seeds_meeting_target']} / {len(SEEDS)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    summaries = [summarize(exp, args.suffix) for exp in EXPERIMENTS]
    for s in summaries:
        print_summary(s)
    out = REPORTS / f"seeds_summary{args.suffix}.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()