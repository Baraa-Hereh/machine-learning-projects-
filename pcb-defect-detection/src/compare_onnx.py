import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from baseline import load_config
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from onnx_detector import OnnxDetector  # noqa: E402

N_IMAGES = 50
SHOW_MIN = 0.3


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final, v2 = cfg["final"], cfg["final"]["v2"]
    raw = root / cfg["data"]["raw_dir"]
    val = pd.read_csv(root / cfg["data"]["splits_dir"] / "val.csv").head(N_IMAGES)
    name = final["experiment"]
    onnx_path = ROOT / f"models/onnx/{name}_seed{v2['seed']}.onnx"

    torch_model, exp = load_model(ROOT / f"exps/{name}.py",
                                  ROOT / f"models/yolox/{name}_seed{v2['seed']}/best_ckpt.pth", "cpu")
    onnx_cpu = OnnxDetector(onnx_path, score_mode=final["score"], providers=["CPUExecutionProvider"])

    count_diff, score_diffs, ious, board_diffs = 0, [], [], []
    t_torch, t_onnx = [], []
    for i, (_, r) in enumerate(val.iterrows()):
        gray = cv2.medianBlur(cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE), 3)

        t0 = time.perf_counter()
        tb, ts = detect(torch_model, exp, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "cpu", final["score"])
        t1 = time.perf_counter()
        ob, os_, _ = onnx_cpu(gray)
        t2 = time.perf_counter()
        if i >= 5:
            t_torch.append((t1 - t0) * 1000)
            t_onnx.append((t2 - t1) * 1000)

        tb, ts = tb[ts >= SHOW_MIN], ts[ts >= SHOW_MIN]
        ob, os_ = ob[os_ >= SHOW_MIN], os_[os_ >= SHOW_MIN]
        count_diff += abs(len(tb) - len(ob))
        board_diffs.append(abs((ts.max() if len(ts) else 0) - (os_.max() if len(os_) else 0)))
        for b, s in zip(tb, ts):
            if len(ob):
                j = int(np.argmax([iou(b, o) for o in ob]))
                ious.append(iou(b, ob[j]))
                score_diffs.append(abs(s - os_[j]))

    print(f"images: {N_IMAGES}, boxes >= {SHOW_MIN}")
    print(f"box count difference (total):   {count_diff}")
    print(f"min IoU torch vs onnx box:      {min(ious):.4f}")
    print(f"max score difference per box:   {max(score_diffs):.5f}")
    print(f"max board score difference:     {max(board_diffs):.5f}")
    print(f"CPU latency p50  torch {np.percentile(t_torch, 50):6.1f} ms   onnx {np.percentile(t_onnx, 50):6.1f} ms")
    print(f"CPU latency p95  torch {np.percentile(t_torch, 95):6.1f} ms   onnx {np.percentile(t_onnx, 95):6.1f} ms")


if __name__ == "__main__":
    main()