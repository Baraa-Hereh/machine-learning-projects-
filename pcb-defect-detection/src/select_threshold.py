from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "pcb_yolox_s_neg_e50"
ENSEMBLE_SEEDS = (0, 1)
SCORE_MODE = "obj"
THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.01), 2)
TARGET_RECALL = 0.995
BAD_REFS = {"77000016", "77000011", "77000022"}


def max_score(model, exp, path, device):
    _, sc = detect(model, exp, cv2.imread(str(path)), device, SCORE_MODE)
    return sc.max() if len(sc) else 0.0


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    defect = np.zeros((len(ENSEMBLE_SEEDS), len(val)))
    clean = np.zeros((len(ENSEMBLE_SEEDS), len(val)))
    for j, s in enumerate(ENSEMBLE_SEEDS):
        ckpt = ROOT / f"models/yolox/{EXP_NAME}_seed{s}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{EXP_NAME}.py", ckpt, device)
        for i, (_, r) in enumerate(val.iterrows()):
            defect[j, i] = max_score(model, exp, raw / r["test_img"], device)
            clean[j, i] = max_score(model, exp, raw / r["ref_img"], device)
        del model
        torch.cuda.empty_cache()

    ids = val["ref_img"].map(lambda p: Path(p).stem.replace("_temp", ""))
    keep = ~ids.isin(BAD_REFS).to_numpy()
    d = defect.mean(axis=0)
    c = clean.mean(axis=0)[keep]
    ref_names = val["ref_img"].to_numpy()[keep]

    table = pd.DataFrame([{"threshold": t,
                           "recall": (d >= t).mean(),
                           "false_alarm": (c >= t).mean(),
                           "n_false_alarm": int((c >= t).sum())}
                          for t in THRESHOLDS])

    ok = table[table["recall"] >= TARGET_RECALL]
    min_fa = ok["false_alarm"].min()
    plateau = ok[ok["false_alarm"] == min_fa]["threshold"]
    chosen = plateau.min()

    print(table.to_string(index=False))
    print(f"\nthresholds with recall >= {TARGET_RECALL} and min FA: "
          f"{plateau.min()} .. {plateau.max()}")
    print(f"chosen threshold (lowest): {chosen}")
    print(f"false-alarm images at chosen: {list(ref_names[c >= chosen])}")


if __name__ == "__main__":
    main()