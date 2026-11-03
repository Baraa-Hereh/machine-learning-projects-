import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    def __init__(self, path, input_size=(640, 640), conf_thre=0.001, nms_thre=0.65,
        score_mode="obj", providers=None):
        available = ort.get_available_providers()
        providers = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thre, self.nms_thre, self.score_mode = conf_thre, nms_thre, score_mode

    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        th, tw = self.input_size
        r = min(th / h, tw / w)
        resized = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((th, tw, 3), 114, np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        return padded.transpose(2, 0, 1)[None].astype(np.float32), r

    def __call__(self, gray):
        x, r = self._preprocess(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        pred = self.session.run(None, {self.input_name: x})[0][0]

        obj = pred[:, 4]
        cls_conf = pred[:, 5:].max(axis=1)
        cls_id = pred[:, 5:].argmax(axis=1)
        conf = obj * cls_conf
        m = conf >= self.conf_thre
        pred, obj, cls_id, conf = pred[m], obj[m], cls_id[m], conf[m]
        if len(pred) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        xywh = np.stack([boxes[:, 0], boxes[:, 1], w, h], axis=1)
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf_thre, self.nms_thre)).flatten()

        scores = obj[idx] if self.score_mode == "obj" else conf[idx]
        return boxes[idx] / r, scores, cls_id[idx]