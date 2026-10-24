import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

from baseline import average_precision, board_score, load_config, match, read_boxes

ROOT = Path(__file__).resolve().parent.parent
CONF_THRE = 0.001
NMS_THRE = 0.65
BOARD_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_model(exp_file, ckpt, device):
    exp = get_exp(str(exp_file), None)
    model = exp.get_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), exp


def detect(model, exp, img, device, score_mode="objcls"):
    x, _ = ValTransform(legacy=False)(img, None, exp.test_size)
    ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = postprocess(model(x), exp.num_classes, CONF_THRE, NMS_THRE, class_agnostic=True)[0]
    if out is None:
        return np.zeros((0, 4)), np.zeros(0)
    out = out.cpu().numpy()
    scores = out[:, 4] if score_mode == "obj" else out[:, 4] * out[:, 5]
    return out[:, :4] / ratio, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=ROOT / "exps/pcb_yolox_s.py")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "models/yolox/pcb_yolox_s/best_ckpt.pth")
    parser.add_argument("--score", choices=["objcls", "obj"], default="objcls")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cfg, root = load_config(ROOT / "configs/config.yaml")
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / f"{args.split}.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, exp = load_model(args.exp, args.ckpt, device)

    results, clean_scores = [], []
    for _, r in val.iterrows():
        boxes, scores = detect(model, exp, cv2.imread(str(raw / r["test_img"])), device, args.score)
        results.append({"boxes": boxes, "scores": scores, "gt": read_boxes(raw / r["ann"])})
        _, ref_scores = detect(model, exp, cv2.imread(str(raw / r["ref_img"])), device, args.score)
        clean_scores.append(board_score(ref_scores))

    n_gt = sum(len(r["gt"]) for r in results)
    all_scores = np.concatenate([r["scores"] for r in results])

    def ap_at(iou_thr):
        is_tp = np.concatenate([match(r["boxes"], r["scores"], r["gt"], iou_thr) for r in results])
        return average_precision(all_scores, is_tp, n_gt)

    ap50 = ap_at(0.5)
    ap50_95 = float(np.mean([ap_at(t) for t in np.arange(0.5, 0.96, 0.05)]))

    defect_scores = np.array([board_score(r["scores"]) for r in results])
    clean_scores = np.array(clean_scores)
    table = [
        {
            "threshold": t,
            "board_recall": float((defect_scores >= t).mean()),
            "board_false_alarm": float((clean_scores >= t).mean()),
        }
        for t in BOARD_THRESHOLDS
    ]

    summary = {
        "split": "val",
        "checkpoint": str(args.ckpt.resolve().relative_to(ROOT)),
        "class_agnostic": True,
        "score": args.score,
        "ap50": ap50,
        "ap50_95": ap50_95,
        "board_table": table,
        "split": args.split,
    }
    suffix = "" if args.score == "objcls" else "_obj"
    out = ROOT / "reports" / f"{args.ckpt.resolve().parent.name}_{args.split}_agnostic{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"[{args.score}] AP@0.5 {ap50:.3f} | AP@0.5:0.95 {ap50_95:.3f}  (class-agnostic, val)")
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()