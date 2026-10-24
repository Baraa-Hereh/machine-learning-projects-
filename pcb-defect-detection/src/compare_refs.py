import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from baseline import load_config
from evaluate_detector import detect, load_model

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "reports/test"
OUT_DIR = TEST_DIR / "compare"
DRAW_MIN = 0.5
PAD = 40
ZOOM = 4


def put_box(img, box, color, text, thick):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if text:
        cv2.putText(img, text, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def crop(img, box):
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = img.shape[:2]
    c = img[max(y1 - PAD, 0):min(y2 + PAD, h), max(x1 - PAD, 0):min(x2 + PAD, w)]
    return cv2.resize(c, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def side_by_side(left, right):
    sep = np.full((left.shape[0], 10, 3), 128, np.uint8)
    return np.hstack([left, sep, right])


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    final = cfg["final"]
    classes = cfg["data"]["classes"]
    raw = root / cfg["data"]["raw_dir"]
    test = pd.read_csv(root / cfg["data"]["splits_dir"] / "test.csv")
    flagged = json.loads((TEST_DIR / "target3_summary.json").read_text())["flagged_refs"]
    rows = test[test["ref_img"].isin(flagged)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds = {name: [] for name in flagged}
    for s in final["ensemble_seeds"]:
        ckpt = ROOT / f"models/yolox/{final['experiment']}_seed{s}/best_ckpt.pth"
        model, exp = load_model(ROOT / f"exps/{final['experiment']}.py", ckpt, device)
        for _, r in rows.iterrows():
            boxes, sc = detect(model, exp, cv2.imread(str(raw / r["ref_img"])), device, final["score"])
            preds[r["ref_img"]] += [(b, f"s{s} {v:.2f}", v) for b, v in zip(boxes, sc) if v >= DRAW_MIN]
        del model
        torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for _, r in rows.iterrows():
        ref = cv2.imread(str(raw / r["ref_img"]))
        tst = cv2.imread(str(raw / r["test_img"]))
        ref_clean, tst_clean = ref.copy(), tst.copy()

        for box, text, _ in preds[r["ref_img"]]:
            put_box(ref, box, (0, 0, 255), text, 2)
            put_box(tst, box, (0, 0, 255), "", 2)
        for lab in np.loadtxt(raw / r["ann"], ndmin=2):
            put_box(tst, lab, (0, 200, 0), classes[int(lab[4]) - 1], 1)

        stem = Path(r["ref_img"]).stem.replace("_temp", "")
        cv2.imwrite(str(OUT_DIR / f"{stem}_full.png"), side_by_side(ref, tst))

        best = max(preds[r["ref_img"]], key=lambda p: p[2])[0]
        cv2.imwrite(str(OUT_DIR / f"{stem}_zoom.png"),
                    side_by_side(crop(ref_clean, best), crop(tst_clean, best)))

    sheet = TEST_DIR / "inspection.csv"
    if not sheet.exists():
        pd.DataFrame({"ref_img": flagged, "case": "", "note": ""}).to_csv(sheet, index=False)
    print(f"saved {len(rows)} comparisons to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()