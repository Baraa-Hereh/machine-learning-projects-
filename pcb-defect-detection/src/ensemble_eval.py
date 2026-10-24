import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "pcb_yolox_s_neg_e50"
SEEDS = range(5)
SCORE_MODE = "obj"
THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.01), 2)
MAX_FA = 0.005
TARGET_RECALL = 0.995
BAD_REFS = {"77000016", "77000011", "77000022"}


def board_scores(val, raw, device):
    defect = np.zeros((len(SEEDS), len(val)))
    clean = np.zeros((len(SEEDS), len(val)))
    for s in SEEDS:
        ckpt = ROOT / f"models/yolox/{EXP_NAME}_seed{s}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{EXP_NAME}.py", ckpt, device)
        for i, (_, r) in enumerate(val.iterrows()):
            _, sc = detect(model, exp, cv2.imread(str(raw / r["test_img"])), device, SCORE_MODE)
            defect[s, i] = sc.max() if len(sc) else 0.0
            _, sc = detect(model, exp, cv2.imread(str(raw / r["ref_img"])), device, SCORE_MODE)
            clean[s, i] = sc.max() if len(sc) else 0.0
        del model
        torch.cuda.empty_cache()
        print(f"seed {s} done")
    return defect, clean


def operating_point(defect, clean):
    best_recall, min_fa = np.nan, np.nan
    for t in THRESHOLDS:
        recall = (defect >= t).mean()
        fa = (clean >= t).mean()
        if fa <= MAX_FA:
            best_recall = recall if np.isnan(best_recall) else max(best_recall, recall)
        if recall >= TARGET_RECALL:
            min_fa = fa if np.isnan(min_fa) else min(min_fa, fa)
    return best_recall, min_fa


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    defect, clean = board_scores(val, raw, device)

    ids = val["ref_img"].map(lambda p: Path(p).stem.replace("_temp", ""))
    masks = {"all": np.ones(len(val), bool), "clean_only": ~ids.isin(BAD_REFS).to_numpy()}

    rows = []
    for k in range(1, len(SEEDS) + 1):
        for combo in itertools.combinations(SEEDS, k):
            d = defect[list(combo)].mean(axis=0)
            c = clean[list(combo)].mean(axis=0)
            for refs, mask in masks.items():
                r, f = operating_point(d, c[mask])
                rows.append({"refs": refs, "k": k, "combo": "-".join(map(str, combo)),
                             "best_recall_at_fa": r, "min_fa_at_recall": f})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/ensemble_val.csv", index=False)

    summary = df.groupby(["refs", "k"]).agg(
        n_combos=("combo", "count"),
        min_fa_mean=("min_fa_at_recall", "mean"),
        min_fa_std=("min_fa_at_recall", "std"),
        best_recall_mean=("best_recall_at_fa", "mean"),
        meeting=("min_fa_at_recall", lambda x: int((x <= MAX_FA).sum())),
    )
    print(summary.round(4).to_string())
    print("saved reports/ensemble_val.csv")


if __name__ == "__main__":
    main()