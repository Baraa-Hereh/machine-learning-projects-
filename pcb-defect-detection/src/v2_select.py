from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config, read_boxes
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports/v2"
SEEDS = (0, 1)
CONFIGS = {"seed0": (0,), "ens01": (0, 1)}
N_GRID = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 80, 120]
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.01), 2)
TARGET_RECALL = 0.995
BAD_REFS = {"77000016", "77000011", "77000022"}


def diff_mask(board, ref, kernel):
    d = cv2.absdiff(board, ref)
    _, m = cv2.threshold(d, 127, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k) > 0


def white_count(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    return int(mask[y1:y2, x1:x2].sum()) if x2 > x1 and y2 > y1 else 0


def touches_gt(box, gt):
    if len(gt) == 0:
        return False
    ix = np.minimum(box[2], gt[:, 2]) - np.maximum(box[0], gt[:, 0])
    iy = np.minimum(box[3], gt[:, 3]) - np.maximum(box[1], gt[:, 1])
    return bool(((ix > 0) & (iy > 0)).any())


def collect(val, raw, final, device):
    masks, gts = [], []
    for _, r in val.iterrows():
        board = cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(raw / r["ref_img"]), cv2.IMREAD_GRAYSCALE)
        masks.append(diff_mask(board, ref, final["baseline_kernel"]))
        gts.append(read_boxes(raw / r["ann"]))

    data = {}
    for s in SEEDS:
        ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{s}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)
        data[s] = []
        for i, (_, r) in enumerate(val.iterrows()):
            boxes, scores = detect(model, exp, cv2.imread(str(raw / r["test_img"])), device, final["score"])
            counts = np.array([white_count(masks[i], b) for b in boxes], dtype=int)
            spurious = np.array([not touches_gt(b, gts[i]) for b in boxes], dtype=bool)
            data[s].append((scores, counts, spurious))
        del model
        torch.cuda.empty_cache()
        print(f"seed {s} done")
    return data


def board_scores(items, n):
    best = np.zeros(len(items))
    spur = np.zeros(len(items))
    for i, (sc, cnt, sp) in enumerate(items):
        keep = cnt >= n
        if keep.any():
            best[i] = sc[keep].max()
        if (keep & sp).any():
            spur[i] = sc[keep & sp].max()
    return best, spur


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids = val["ref_img"].map(lambda p: Path(p).stem.replace("_temp", ""))
    clean = ~ids.isin(BAD_REFS).to_numpy()
    data = collect(val, raw, final, device)

    rows = []
    for name, seeds in CONFIGS.items():
        for n in N_GRID:
            per = [board_scores(data[s], n) for s in seeds]
            best = np.mean([p[0] for p in per], axis=0)
            spur = np.mean([p[1] for p in per], axis=0)
            for t in THRESHOLDS:
                rows.append({"config": name, "N": n, "threshold": t,
                             "recall": float((best >= t).mean()),
                             "n_false_alarm": int((spur[clean] >= t).sum())})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v2_val_grid.csv", index=False)

    for name in CONFIGS:
        ok = df[(df["config"] == name) & (df["recall"] >= TARGET_RECALL)]
        m = ok["n_false_alarm"].min()
        print(f"\n=== {name} ===  min false alarms with recall >= {TARGET_RECALL}: {m} / {clean.sum()}")
        for n, g in ok[ok["n_false_alarm"] == m].groupby("N"):
            print(f"  N={n:<4} thresholds {g['threshold'].min():.2f} .. {g['threshold'].max():.2f}")
        v1 = ok[ok["N"] == 0]["n_false_alarm"].min()
        print(f"  (N=0, no reference check: {v1})")


if __name__ == "__main__":
    main()