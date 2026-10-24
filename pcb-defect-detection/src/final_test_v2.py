import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config, read_boxes
from diagnose_errors import draw
from evaluate_detector import detect, load_model
from v2_pipeline import prepare
from v2_select import diff_mask, touches_gt, white_count

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports/test_v2"
BAD_TEST_REFS = {"12100185", "50600055"}


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    v2 = final["v2"]
    raw = root / cfg["data"]["raw_dir"]
    test = pd.read_csv(root / cfg["data"]["splits_dir"] / "test.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thr, n_min = v2["board_threshold"], v2["min_white_pixels"]

    ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{v2['seed']}/best_ckpt.pth"
    model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)

    best = np.zeros(len(test))
    spur = np.zeros(len(test))
    spur_boxes = {}
    for i, (_, r) in enumerate(test.iterrows()):
        board = cv2.imread(str(raw / r["test_img"]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(raw / r["ref_img"]), cv2.IMREAD_GRAYSCALE)
        board, ref, (ox, oy) = prepare(board, ref)
        gt = read_boxes(raw / r["ann"]).astype(float)
        if len(gt):
            gt[:, [0, 2]] += ox
            gt[:, [1, 3]] += oy

        mask = diff_mask(board, ref, final["baseline_kernel"])
        boxes, scores = detect(model, exp, cv2.cvtColor(board, cv2.COLOR_GRAY2BGR), device, final["score"])
        for b, s in zip(boxes, scores):
            if white_count(mask, b) < n_min:
                continue
            best[i] = max(best[i], s)
            if not touches_gt(b, gt):
                spur[i] = max(spur[i], s)
                if s >= thr:
                    spur_boxes.setdefault(i, []).append((b, f"{s:.2f}"))

    ids = test["ref_img"].map(lambda p: Path(p).stem.replace("_temp", ""))
    clean = ~ids.isin(BAD_TEST_REFS).to_numpy()
    missed = np.where(best < thr)[0]
    flagged = np.where(spur >= thr)[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in flagged:
        r = test.iloc[i]
        draw(raw / r["test_img"], spur_boxes[i], [], OUT_DIR / "spurious" / f"{Path(r['test_img']).stem}.png")

    summary = {
        "split": "test (seen before, v1)",
        "seed": v2["seed"], "N": n_min, "threshold": thr,
        "board_recall": float((best >= thr).mean()), "n_missed": int(len(missed)),
        "spurious_fa_all": float((spur >= thr).mean()), "n_spurious_all": int(len(flagged)),
        "spurious_fa_excl": float((spur[clean] >= thr).mean()),
        "n_spurious_excl": int((spur[clean] >= thr).sum()), "n_excl_boards": int(clean.sum()),
        "missed_boards": test["test_img"].iloc[missed].tolist(),
        "spurious_boards": test["test_img"].iloc[flagged].tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"board recall:      {summary['board_recall']:.4f}  ({len(test) - len(missed)} / {len(test)})")
    print(f"spurious FA (all): {summary['spurious_fa_all']:.4f}  ({len(flagged)} / {len(test)})")
    print(f"spurious FA (excl):{summary['spurious_fa_excl']:.4f}  "
          f"({summary['n_spurious_excl']} / {summary['n_excl_boards']})")
    print(f"missed:   {summary['missed_boards']}")
    print(f"spurious: {summary['spurious_boards']}")


if __name__ == "__main__":
    main()