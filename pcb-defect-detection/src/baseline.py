import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent.parent
    return cfg, root


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_boxes(path):
    return np.loadtxt(path, dtype=int, ndmin=2)[:, :4]


def detect(test, ref, kernel_size):
    diff = cv2.absdiff(test, ref)
    _, mask = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = stats[1:, :4].copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    scores = stats[1:, 4].astype(float)
    return boxes, scores


def add_margin(boxes, margin, size):
    out = boxes.copy()
    out[:, :2] -= margin
    out[:, 2:] += margin
    return np.clip(out, 0, size)


def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1)


def match(pred, scores, gt, iou_thr):
    is_tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return is_tp
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), bool)
    for i in np.argsort(-scores):
        candidates = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_thr:
            is_tp[i] = True
            taken[j] = True
    return is_tp


def average_precision(scores, is_tp, n_gt):
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = np.concatenate([[0.0], tp / n_gt, [1.0]])
    precision = np.concatenate([[0.0], tp / np.maximum(tp + fp, 1), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def dataset_ap(results, margin, iou_thr, size):
    all_scores, all_tp, n_gt = [], [], 0
    for r in results:
        pred = add_margin(r["boxes"], margin, size)
        all_scores.append(r["scores"])
        all_tp.append(match(pred, r["scores"], r["gt"], iou_thr))
        n_gt += len(r["gt"])
    return average_precision(np.concatenate(all_scores), np.concatenate(all_tp), n_gt)


def best_shift(test, ref, max_shift):
    m = max_shift
    h, w = test.shape
    t = test[m:h - m, m:w - m]
    best, best_count = (0, 0), None
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            r = ref[m + dy:h - m + dy, m + dx:w - m + dx]
            count = np.count_nonzero(cv2.absdiff(t, r) > 127)
            if best_count is None or count < best_count:
                best, best_count = (dx, dy), count
    return best


def train_shifts(cfg, root):
    path = root / cfg["baseline"]["shifts_file"]
    if path.exists():
        return pd.read_csv(path)
    raw = root / cfg["data"]["raw_dir"]
    train = pd.read_csv(root / cfg["data"]["splits_dir"] / "train.csv")
    rows = []
    for _, r in train.iterrows():
        dx, dy = best_shift(
            read_gray(raw / r["test_img"]),
            read_gray(raw / r["ref_img"]),
            cfg["baseline"]["max_shift_search"],
        )
        rows.append({"id": r["id"], "dx": dx, "dy": dy})
    shifts = pd.DataFrame(rows)
    shifts.to_csv(path, index=False)
    return shifts


def shifted_pair(ref, dx, dy, pad):
    h, w = ref.shape
    base = ref[pad:h - pad, pad:w - pad]
    moved = ref[pad + dy:h - pad + dy, pad + dx:w - pad + dx]
    return moved, base


def board_score(scores):
    return float(scores.max()) if len(scores) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/config.yaml",
    )
    args = parser.parse_args()
    cfg, root = load_config(args.config)
    bcfg = cfg["baseline"]
    size = cfg["data"]["image_size"]
    raw = root / cfg["data"]["raw_dir"]

    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    pairs = [
        (read_gray(raw / r["test_img"]), read_gray(raw / r["ref_img"]), read_boxes(raw / r["ann"]))
        for _, r in val.iterrows()
    ]

    grid, detections = [], {}
    for k in bcfg["kernel_sizes"]:
        results = []
        for test, ref, gt in pairs:
            boxes, scores = detect(test, ref, k)
            results.append({"boxes": boxes, "scores": scores, "gt": gt})
        detections[k] = results
        for m in bcfg["margins"]:
            ap50 = dataset_ap(results, m, 0.5, size)
            grid.append({"kernel": k, "margin": m, "ap50": ap50})
            print(f"kernel {k} margin {m:2d}: AP@0.5 {ap50:.3f}")

    grid_df = pd.DataFrame(grid)
    best = grid_df.loc[grid_df["ap50"].idxmax()]
    k, m = int(best["kernel"]), int(best["margin"])
    results = detections[k]

    ap_range = [dataset_ap(results, m, t, size) for t in np.arange(0.5, 0.96, 0.05)]
    defect_scores = np.array([board_score(r["scores"]) for r in results])

    shifts = train_shifts(cfg, root)
    limit = bcfg["max_shift_valid"]
    valid = shifts[(shifts["dx"].abs() <= limit) & (shifts["dy"].abs() <= limit)]
    rng = np.random.default_rng(bcfg["seed"])
    picks = valid.iloc[rng.integers(0, len(valid), len(pairs))]

    clean_scores = []
    for (_, ref, _), (_, s) in zip(pairs, picks.iterrows()):
        moved, base = shifted_pair(ref, int(s["dx"]), int(s["dy"]), limit)
        _, scores = detect(moved, base, k)
        clean_scores.append(board_score(scores))
    clean_scores = np.array(clean_scores)

    table = [
        {
            "threshold": t,
            "board_recall": float((defect_scores >= t).mean()),
            "board_false_alarm": float((clean_scores >= t).mean()),
        }
        for t in bcfg["board_thresholds"]
    ]

    summary = {
        "split": "val",
        "kernel": k,
        "margin": m,
        "ap50": float(best["ap50"]),
        "ap50_95": float(np.mean(ap_range)),
        "simulated_shift_share_nonzero": float(((picks["dx"] != 0) | (picks["dy"] != 0)).mean()),
        "board_table": table,
    }

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    grid_df.to_csv(reports / "baseline_val_grid.csv", index=False)
    with open(reports / "baseline_val.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nbest: kernel {k}, margin {m}")
    print(f"AP@0.5 {summary['ap50']:.3f} | AP@0.5:0.95 {summary['ap50_95']:.3f}")
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()