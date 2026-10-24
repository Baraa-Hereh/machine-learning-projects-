import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

from baseline import load_config, read_boxes
from evaluate_detector import detect, load_model
from v2_pipeline import prepare
from v2_select import diff_mask, white_count

ROOT = Path(__file__).resolve().parent.parent


def main(name):
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final, v2 = cfg["final"], cfg["final"]["v2"]
    raw = root / cfg["data"]["raw_dir"]
    test = pd.read_csv(root / cfg["data"]["splits_dir"] / "test.csv")
    r = test[test["test_img"].str.contains(name)].iloc[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{v2['seed']}/best_ckpt.pth"
    model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)

    board = cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(raw / r["ref_img"]), cv2.IMREAD_GRAYSCALE)
    board, ref, (ox, oy) = prepare(board, ref)
    mask = diff_mask(board, ref, final["baseline_kernel"])
    boxes, scores = detect(model, exp, cv2.cvtColor(board, cv2.COLOR_GRAY2BGR), device, final["score"])

    print(f"offset {ox, oy}")
    for j, g in enumerate(read_boxes(raw / r["ann"])):
        g = g[:4] + [ox, oy, ox, oy]
        best_s, best_w = 0.0, 0
        for b, s in zip(boxes, scores):
            if b[0] < g[2] and b[2] > g[0] and b[1] < g[3] and b[3] > g[1] and s > best_s:
                best_s, best_w = s, white_count(mask, b)
        print(f"defect {j}: box {g.astype(int).tolist()}  white in label {white_count(mask, g)}"
              f"  best score {best_s:.2f}  white in that box {best_w}")


if __name__ == "__main__":
    main(sys.argv[1])