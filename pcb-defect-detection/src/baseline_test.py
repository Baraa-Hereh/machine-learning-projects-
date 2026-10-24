import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import dataset_ap, detect, load_config, read_boxes, read_gray

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    raw = root / cfg["data"]["raw_dir"]
    size = cfg["data"]["image_size"]
    test = pd.read_csv(root / cfg["data"]["splits_dir"] / "test.csv")

    results = []
    for _, r in test.iterrows():
        boxes, scores = detect(read_gray(raw / r["test_img"]), read_gray(raw / r["ref_img"]),
                               final["baseline_kernel"])
        results.append({"boxes": boxes, "scores": scores, "gt": read_boxes(raw / r["ann"])})

    margin = final["baseline_margin"]
    ap50 = dataset_ap(results, margin, 0.5, size)
    ap50_95 = float(np.mean([dataset_ap(results, margin, t, size)
                             for t in np.arange(0.5, 0.96, 0.05)]))

    out = ROOT / "reports/test/baseline_test.json"
    out.write_text(json.dumps({"split": "test", "kernel": final["baseline_kernel"],
                               "margin": margin, "ap50": ap50, "ap50_95": ap50_95}, indent=2))
    print(f"baseline test  AP50 {ap50:.3f} | AP50:95 {ap50_95:.3f}")


if __name__ == "__main__":
    main()