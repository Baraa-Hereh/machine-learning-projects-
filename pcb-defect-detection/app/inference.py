from pathlib import Path

import cv2
import numpy as np
import yaml

from onnx_detector import OnnxDetector
from pipeline import diff_mask, prepare, white_count

APP_DIR = Path(__file__).resolve().parent
NO_REF_WARNING = "no reference for this board; reference check skipped, higher false-alarm rate"


class Inspector:
    def __init__(self, settings_path=APP_DIR / "settings.yaml"):
        self.s = yaml.safe_load(Path(settings_path).read_text())
        self.reference_dir = (APP_DIR / self.s["references"]).resolve()
        self.detector = OnnxDetector(APP_DIR / self.s["model"], tuple(self.s["input_size"]),
                                     self.s["conf_thre"], self.s["nms_thre"], self.s["score"])
        self.device = self.detector.session.get_providers()[0]

    def load_reference(self, reference_id):
        path = self.reference_dir / f"{reference_id}.jpg"
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None

    def inspect(self, board, reference=None):
        h, w = board.shape
        if reference is not None:
            if reference.shape != board.shape:
                raise ValueError(f"reference size {reference.shape} != image size {board.shape}")
            img, ref, (ox, oy) = prepare(board, reference)
            threshold, warning = self.s["board_threshold"], None
        else:
            img, ref, (ox, oy) = cv2.medianBlur(board, 3), None, (0, 0)
            threshold, warning = self.s["fallback_threshold"], NO_REF_WARNING

        boxes, scores, classes = self.detector(img)
        if ref is not None:
            mask = diff_mask(img, ref, self.s["diff_kernel"])
            keep = np.array([white_count(mask, b) >= self.s["min_white_pixels"] for b in boxes], bool)
        else:
            keep = np.ones(len(boxes), bool)

        board_score = float(scores[keep].max()) if keep.any() else 0.0
        defects, candidates = [], []
        for b, s, c in zip(boxes[keep], scores[keep], classes[keep]):
            if s < self.s["candidate_threshold"]:
                continue
            x1, y1, x2, y2 = b - np.array([ox, oy, ox, oy])
            item = {"type": self.s["classes"][c], "score": round(float(s), 3),
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