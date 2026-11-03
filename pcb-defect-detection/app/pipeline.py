import cv2
import numpy as np


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def align(board, ref):
    (sx, sy), _ = cv2.phaseCorrelate(ref.astype(np.float32), board.astype(np.float32))
    candidates = [(-round(sx), -round(sy)), (round(sx), round(sy)), (0, 0)]
    best = min(candidates, key=lambda d: np.count_nonzero(shift(board, *d) != ref))
    return shift(board, *best), best


def prepare(board, ref):
    board = cv2.medianBlur(board, 3)
    ref = cv2.medianBlur(ref, 3)
    board, offset = align(board, ref)
    return board, ref, offset


def diff_mask(board, ref, kernel):
    d = cv2.absdiff(board, ref)
    _, m = cv2.threshold(d, 127, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k) > 0


def white_count(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    return int(mask[y1:y2, x1:x2].sum()) if x2 > x1 and y2 > y1 else 0