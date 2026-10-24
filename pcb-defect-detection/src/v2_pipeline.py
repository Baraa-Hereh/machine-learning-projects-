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