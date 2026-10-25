import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config
from evaluate_detector import detect, load_model
from v2_pipeline import prepare
from v2_select import diff_mask, white_count

ROOT = Path(__file__).resolve().parent.parent
WARMUP = 10


def now(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final, v2 = cfg["final"], cfg["final"]["v2"]
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{v2['seed']}/best_ckpt.pth"
    model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)

    times = {"prepare": [], "detect": [], "check": [], "total": []}
    for i, (_, r) in enumerate(val.iterrows()):
        board = cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(raw / r["ref_img"]), cv2.IMREAD_GRAYSCALE)

        t0 = now(device)
        b, rf, _ = prepare(board, ref)
        t1 = now(device)
        boxes, scores = detect(model, exp, cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), device, final["score"])
        t2 = now(device)
        mask = diff_mask(b, rf, final["baseline_kernel"])
        _ = [white_count(mask, x) for x in boxes]
        t3 = now(device)

        if i >= WARMUP:
            for k, v in zip(times, [t1 - t0, t2 - t1, t3 - t2, t3 - t0]):
                times[k].append(v * 1000)

    print(f"device: {device}, boards: {len(times['total'])}")
    for k, v in times.items():
        print(f"{k:<8} p50 {np.percentile(v, 50):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms")


if __name__ == "__main__":
    main()