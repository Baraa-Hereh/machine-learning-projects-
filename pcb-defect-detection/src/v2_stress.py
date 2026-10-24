import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config, read_boxes
from evaluate_detector import detect, load_model
from v2_pipeline import prepare, shift
from v2_select import (BAD_REFS, N_GRID, THRESHOLDS, TARGET_RECALL,
                       board_scores, diff_mask, touches_gt, white_count)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports/v2"
SEED = 0
RNG_SEED = 42
SCENARIOS = [
    ("none", 0, 0, 0.0),
    ("shift1", 1, 1, 0.0),
    ("shift2", 2, 2, 0.0),
    ("shift3", 3, 3, 0.0),
    ("noise", 0, 0, 0.002),
    ("shift2+noise", 2, 2, 0.002),
]


def add_noise(img, p, rng):
    if p == 0:
        return img
    out = img.copy()
    flip = rng.random(img.shape) < p
    out[flip] = 255 - out[flip]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = val["ref_img"].map(lambda p: Path(p).stem.replace("_temp", ""))
    clean = ~ids.isin(BAD_REFS).to_numpy()

    ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{SEED}/best_ckpt.pth"
    model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)

    rows = []
    for name, dx, dy, p in SCENARIOS:
        rng = np.random.default_rng(RNG_SEED)
        items = []
        for _, r in val.iterrows():
            board = cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(raw / r["ref_img"]), cv2.IMREAD_GRAYSCALE)
            board = add_noise(shift(board, dx, dy), p, rng)

            gt = read_boxes(raw / r["ann"]).astype(float)
            if len(gt):
                gt[:, [0, 2]] += dx
                gt[:, [1, 3]] += dy

            if args.prepare:
                board, ref, (ox, oy) = prepare(board, ref)
                if len(gt):
                    gt[:, [0, 2]] += ox
                    gt[:, [1, 3]] += oy

            mask = diff_mask(board, ref, final["baseline_kernel"])
            boxes, scores = detect(model, exp, cv2.cvtColor(board, cv2.COLOR_GRAY2BGR),
                                   device, final["score"])
            counts = np.array([white_count(mask, b) for b in boxes], dtype=int)
            spurious = np.array([not touches_gt(b, gt) for b in boxes], dtype=bool)
            items.append((scores, counts, spurious))

        for n in N_GRID:
            best, spur = board_scores(items, n)
            for t in THRESHOLDS:
                rows.append({"scenario": name, "N": n, "threshold": t,
                             "recall": float((best >= t).mean()),
                             "n_false_alarm": int((spur[clean] >= t).sum())})
        print(f"{name} done")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    suffix = "_prepared" if args.prepare else ""
    df.to_csv(OUT_DIR / f"v2_stress_grid{suffix}.csv", index=False)

    for name, *_ in SCENARIOS:
        print(f"\n=== {name} ===")
        sub = df[(df["scenario"] == name) & (df["recall"] >= TARGET_RECALL)]
        for n, g in sub.groupby("N"):
            m = g["n_false_alarm"].min()
            best = g[g["n_false_alarm"] == m]["threshold"]
            print(f"  N={n:<4} min FA {m:>3} / {clean.sum()}   thresholds {best.min():.2f} .. {best.max():.2f}")

    worst = df.groupby(["N", "threshold"]).agg(
        worst_recall=("recall", "min"), worst_fa=("n_false_alarm", "max")).reset_index()
    ok = worst[worst["worst_recall"] >= TARGET_RECALL]
    print("\n=== robust (worst case over all scenarios) ===")
    for n, g in ok.groupby("N"):
        m = g["worst_fa"].min()
        best = g[g["worst_fa"] == m]["threshold"]
        print(f"  N={n:<4} worst FA {m:>3} / {clean.sum()}   thresholds {best.min():.2f} .. {best.max():.2f}")


if __name__ == "__main__":
    main()