from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config, read_boxes
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "pcb_yolox_s_neg_e50"
SEEDS = range(5)
FP_THR = 0.8
FN_THR = 0.9
IOU_THR = 0.5
MIN_SEEDS = 3
OUT_DIR = ROOT / "reports/diagnosis"


def iou_with(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter)


def run_seeds(val, raw, device):
    test_dets, ref_dets = [], []
    for seed in SEEDS:
        ckpt = ROOT / f"models/yolox/{EXP_NAME}_seed{seed}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{EXP_NAME}.py", ckpt, device)
        test_dets.append([detect(model, exp, cv2.imread(str(raw / r["test_img"])), device)
                          for _, r in val.iterrows()])
        ref_dets.append([detect(model, exp, cv2.imread(str(raw / r["ref_img"])), device)
                         for _, r in val.iterrows()])
        del model
        torch.cuda.empty_cache()
        print(f"seed {seed} done")
    return test_dets, ref_dets


def false_alarms(val, ref_dets):
    rows = []
    for i, (_, r) in enumerate(val.iterrows()):
        max_scores = [scores.max() if len(scores) else 0.0
                      for _, scores in (ref_dets[s][i] for s in SEEDS)]
        rows.append({"ref_img": r["ref_img"],
                     "n_seeds": int(sum(m >= FP_THR for m in max_scores)),
                     "mean_max_score": float(np.mean(max_scores))})
    return pd.DataFrame(rows).sort_values("n_seeds", ascending=False)


def missed_defects(val, raw, test_dets):
    rows = []
    for i, (_, r) in enumerate(val.iterrows()):
        gt = read_boxes(raw / r["ann"])
        for j, g in enumerate(gt):
            missed, board_missed, best = 0, 0, []
            for s in SEEDS:
                boxes, scores = test_dets[s][i]
                ious = iou_with(g, boxes) if len(boxes) else np.zeros(0)
                overlap = scores[ious >= IOU_THR]
                best.append(overlap.max() if len(overlap) else 0.0)
                missed += best[-1] < FN_THR
                board_missed += (scores.max() if len(scores) else 0.0) < FN_THR
            rows.append({"test_img": r["test_img"], "box_id": j,
                         "cls": int(g[4]) if len(g) > 4 else -1,
                         "w": float(g[2] - g[0]), "h": float(g[3] - g[1]),
                         "n_seeds_missed": int(missed),
                         "n_seeds_board_missed": int(board_missed),
                         "mean_best_score": float(np.mean(best))})
    return pd.DataFrame(rows).sort_values("n_seeds_missed", ascending=False)


def draw(img_path, red_boxes, green_boxes, out_path):
    img = cv2.imread(str(img_path))
    for box, label in green_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    for box, label in red_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dets, ref_dets = run_seeds(val, raw, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = false_alarms(val, ref_dets)
    fn = missed_defects(val, raw, test_dets)
    fp.to_csv(OUT_DIR / "false_alarms.csv", index=False)
    fn.to_csv(OUT_DIR / "missed_defects.csv", index=False)

    index = {r["ref_img"]: i for i, (_, r) in enumerate(val.iterrows())}
    for _, row in fp[fp["n_seeds"] >= MIN_SEEDS].iterrows():
        i = index[row["ref_img"]]
        red = [(b, f"s{s} {sc:.2f}") for s in SEEDS
               for b, sc in zip(*ref_dets[s][i]) if sc >= FP_THR]
        draw(raw / row["ref_img"], red, [], OUT_DIR / "fp" / f"{Path(row['ref_img']).stem}.png")

    index = {r["test_img"]: i for i, (_, r) in enumerate(val.iterrows())}
    for _, row in fn[fn["n_seeds_missed"] >= MIN_SEEDS].iterrows():
        i = index[row["test_img"]]
        gt = read_boxes(raw / val.iloc[i]["ann"])[row["box_id"]]
        green = [(b, f"{sc:.2f}") for b, sc in zip(*test_dets[0][i]) if sc >= 0.3]
        draw(raw / row["test_img"], [(gt, f"missed {row['mean_best_score']:.2f}")], green,
             OUT_DIR / "fn" / f"{Path(row['test_img']).stem}_box{row['box_id']}.png")

    print(f"false alarms (>= {MIN_SEEDS} seeds): {(fp['n_seeds'] >= MIN_SEEDS).sum()} / {len(fp)}")
    print(f"missed defects (>= {MIN_SEEDS} seeds): {(fn['n_seeds_missed'] >= MIN_SEEDS).sum()} / {len(fn)}")
    print(f"saved to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()