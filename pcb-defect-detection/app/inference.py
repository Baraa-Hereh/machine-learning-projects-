import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline import load_config  # noqa: E402
from evaluate_detector import CONF_THRE, NMS_THRE, load_model  # noqa: E402
from v2_pipeline import prepare  # noqa: E402
from v2_select import diff_mask, white_count  # noqa: E402

NO_REF_WARNING = "no reference for this board; reference check skipped, higher false-alarm rate"


class Inspector:
    def __init__(self, reference_dir=ROOT / "references"):
        cfg, _ = load_config(ROOT / "configs/config.yaml")
        self.final = cfg["final"]
        self.v2 = self.final["v2"]
        self.classes = cfg["data"]["classes"]
        self.reference_dir = Path(reference_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        name = self.final["experiment"]
        ckpt = ROOT / f"models/yolox/{name}_seed{self.v2['seed']}/best_ckpt.pth"
        self.model, self.exp = load_model(ROOT / f"exps/{name}.py", ckpt, self.device)

    def load_reference(self, reference_id):
        path = self.reference_dir / f"{reference_id}.jpg"
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None

    def _detect(self, gray):
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x, _ = ValTransform(legacy=False)(img, None, self.exp.test_size)
        ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
        x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = postprocess(self.model(x), self.exp.num_classes, CONF_THRE, NMS_THRE,
                              class_agnostic=True)[0]
        if out is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        out = out.cpu().numpy()
        scores = out[:, 4] if self.final["score"] == "obj" else out[:, 4] * out[:, 5]
        return out[:, :4] / ratio, scores, out[:, 6].astype(int)

    def inspect(self, board, reference=None):
        h, w = board.shape
        if reference is not None:
            if reference.shape != board.shape:
                raise ValueError(f"reference size {reference.shape} != image size {board.shape}")
            img, ref, (ox, oy) = prepare(board, reference)
            threshold, warning = self.v2["board_threshold"], None
        else:
            img, ref, (ox, oy) = cv2.medianBlur(board, 3), None, (0, 0)
            threshold, warning = self.v2["fallback_threshold"], NO_REF_WARNING

        boxes, scores, classes = self._detect(img)
        if ref is not None:
            mask = diff_mask(img, ref, self.final["baseline_kernel"])
            keep = np.array([white_count(mask, b) >= self.v2["min_white_pixels"] for b in boxes], bool)
        else:
            keep = np.ones(len(boxes), bool)

        board_score = float(scores[keep].max()) if keep.any() else 0.0
        defects, candidates = [], []
        for b, s, c in zip(boxes[keep], scores[keep], classes[keep]):
            if s < self.v2["candidate_threshold"]:
                continue
            x1, y1, x2, y2 = b - np.array([ox, oy, ox, oy])
            item = {"type": self.classes[c], "score": round(float(s), 3),
                    "box": [int(max(x1, 0)), int(max(y1, 0)), int(min(x2, w)), int(min(y2, h))]}
            (defects if s >= threshold else candidates).append(item)

        return {"defective": board_score >= threshold, "board_score": round(board_score, 3),
                "threshold": threshold, "reference_used": ref is not None,
                "warning": warning, "defects": defects, "candidates": candidates}

    def annotate(self, board, result):
        img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        styles = [("defects", (0, 0, 255), 2), ("candidates", (0, 165, 255), 1)]
        for key, color, thick in styles:
            for d in result[key]:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
                cv2.putText(img, f"{d['type']} {d['score']:.2f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(img, "red: defect   orange: check manually", (5, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        return img