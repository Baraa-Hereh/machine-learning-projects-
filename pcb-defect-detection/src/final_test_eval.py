import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config
from diagnose_errors import draw
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports/test"
DRAW_MIN = 0.5


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    raw = root / cfg["data"]["raw_dir"]
    test = pd.read_csv(root / cfg["data"]["splits_dir"] / "test.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thr = final["board_threshold"]
    seeds = final["ensemble_seeds"]
    exp_name = final["experiment"]

    defect = np.zeros((len(seeds), len(test)))
    clean = np.zeros((len(seeds), len(test)))
    ref_dets = [[None] * len(test) for _ in seeds]
    for j, s in enumerate(seeds):
        ckpt = ROOT / f"models/yolox/{exp_name}_seed{s}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{exp_name}.py", ckpt, device)
        for i, (_, r) in enumerate(test.iterrows()):
            _, sc = detect(model, exp, cv2.imread(str(raw / r["test_img"])), device, final["score"])
            defect[j, i] = sc.max() if len(sc) else 0.0
            boxes, sc = detect(model, exp, cv2.imread(str(raw / r["ref_img"])), device, final["score"])
            clean[j, i] = sc.max() if len(sc) else 0.0
            ref_dets[j][i] = (boxes, sc)
        del model
        torch.cuda.empty_cache()
        print(f"seed {s} done")

    d = defect.mean(axis=0)
    c = clean.mean(axis=0)
    flagged = np.where(c >= thr)[0]
    missed = np.where(d < thr)[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"test_img": test["test_img"], "ref_img": test["ref_img"],
                  "defect_score": d, "clean_score": c}).to_csv(OUT_DIR / "board_scores.csv", index=False)

    for i in flagged:
        r = test.iloc[i]
        red = [(b, f"s{s} {sc:.2f}") for j, s in enumerate(seeds)
               for b, sc in zip(*ref_dets[j][i]) if sc >= DRAW_MIN]
        draw(raw / r["ref_img"], red, [], OUT_DIR / "flagged_refs" / f"{Path(r['ref_img']).stem}.png")

    summary = {
        "split": "test",
        "experiment": exp_name,
        "ensemble_seeds": seeds,
        "score": final["score"],
        "board_threshold": thr,
        "n_boards": int(len(test)),
        "board_recall": float((d >= thr).mean()),
        "n_missed": int(len(missed)),
        "board_false_alarm_all_refs": float((c >= thr).mean()),
        "n_false_alarm_all_refs": int(len(flagged)),
        "flagged_refs": test["ref_img"].iloc[flagged].tolist(),
        "missed_boards": test["test_img"].iloc[missed].tolist(),
    }
    (OUT_DIR / "target3_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"board recall:      {summary['board_recall']:.4f}  "
          f"({len(test) - len(missed)} / {len(test)})")
    print(f"board false alarm: {summary['board_false_alarm_all_refs']:.4f}  "
          f"({len(flagged)} / {len(test)})  [all refs, before inspection]")
    print(f"flagged refs:  {summary['flagged_refs']}")
    print(f"missed boards: {summary['missed_boards']}")


if __name__ == "__main__":
    main()