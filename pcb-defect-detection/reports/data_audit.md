# Data Audit — DeepPCB

**Date:** 11 October 2026
**Notebook:** `notebooks/01_data_audit.ipynb` · **Split script:** `src/split.py`

## 1. Summary

The data is complete, well-formed and balanced across defect types. Two findings shape the project: the official test set is mostly board designs never seen in training, so validation must also be held out by design; and every inspection image contains defects, so board-level false alarms must be measured on the defect-free reference images.

## 2. Source

- **DeepPCB** (MIT licence), `github.com/tangsanli5201/DeepPCB`.
- 1,500 image pairs: a 640×640 board region with defects (`*_test.jpg`) and its defect-free reference (`*_temp.jpg`), plus one annotation file per pair.
- Annotation format: one line per defect, `x1 y1 x2 y2 class` (corner format, pixels), classes 1–6.

| ID | Class |
|---|---|
| 1 | open |
| 2 | short |
| 3 | mousebite |
| 4 | spur |
| 5 | spurious copper |
| 6 | pin-hole |

## 3. Integrity

| Check | Result |
|---|---|
| Inspection images | 1,500 ✔ |
| Annotation files | 1,500 ✔ |
| Reference images | 1,501 — one orphan (`group90100/90100/90100034_temp.jpg`) with no inspection image or annotation; not in any official list, excluded |
| Files listed in the official splits that are missing | 0 ✔ |
| Images in both official splits | 0 ✔ |
| Malformed annotation lines (not 5 integers, box outside the image or empty, class outside 1–6) | 0 of 10,013 ✔ |

**Note:** the official lists (`trainval.txt`, `test.txt`) give image names without the `_test` / `_temp` suffix; the split script adds them.

## 4. Images

- All images are **binary** (black and white only), already thresholded by the dataset authors.
- A real line camera produces grayscale images with lighting variation and noise. Client images will be harder than DeepPCB; a pilot on client images is required before deployment.

## 5. Groups are board designs

The 1,500 images come from 11 groups. Viewing one reference image per group confirmed that **each group is a different board design** (track widths, pad shapes, density).

| Group | Official trainval | Official test |
|---|---|---|
| group00041 | 200 | 21 |
| group12000 | 0 | 14 |
| group12100 | 0 | 146 |
| group12300 | 0 | 98 |
| group13000 | 198 | 18 |
| group20085 | 291 | 34 |
| group44000 | 50 | 50 |
| group50600 | 50 | 29 |
| group77000 | 100 | 7 |
| group90100 | 0 | 74 |
| group92000 | 111 | 9 |

Four groups (332 of 500 test images) appear only in the test set. The official test therefore mostly measures performance on **unseen designs**, which matches the client's situation.

## 6. Defects

**10,013 defects** in total.

| Group | copper | mousebite | open | pin-hole | short | spur | Total |
|---|---|---|---|---|---|---|---|
| group00041 | 245 | 372 | 304 | 236 | 212 | 273 | 1,642 |
| group12000 | 14 | 11 | 23 | 11 | 19 | 15 | 93 |
| group12100 | 143 | 177 | 178 | 145 | 131 | 145 | 919 |
| group12300 | 89 | 112 | 137 | 96 | 80 | 86 | 600 |
| group13000 | 255 | 328 | 211 | 254 | 141 | 262 | 1,451 |
| group20085 | 317 | 460 | 508 | 275 | 474 | 394 | 2,428 |
| group44000 | 99 | 147 | 160 | 110 | 120 | 111 | 747 |
| group50600 | 57 | 72 | 85 | 76 | 70 | 72 | 432 |
| group77000 | 90 | 107 | 89 | 113 | 76 | 83 | 558 |
| group90100 | 70 | 76 | 109 | 73 | 88 | 67 | 483 |
| group92000 | 95 | 103 | 138 | 112 | 95 | 117 | 660 |
| **All** | **1,474** | **1,965** | **1,942** | **1,501** | **1,506** | **1,625** | **10,013** |

- **Classes are balanced:** largest to smallest ≈ 1.3 : 1. No class weighting is needed.
- **Every inspection image has at least one defect** (min 1, median 7, max 15, mean 6.7). There are no defect-free inspection images, so the board false-alarm rate must be measured on the reference images.
- **Box size:** median 35 × 33 px (about 5 % of the image width), smallest 18 px, largest 173 px. Defects are mostly small with a long tail of larger ones; the detector needs a fine output grid (stride 8).

## 7. Split

| Split | Groups | Images | Defects |
|---|---|---|---|
| Train | 00041, 13000, 20085, 92000 | 800 | 5,688 |
| Validation | 44000, 50600, 77000 | 200 | 1,185 |
| Test (official, untouched) | all 11 | 500 | 3,140 |

Defects per class and split:

| Split | copper | mousebite | open | pin-hole | short | spur | Total |
|---|---|---|---|---|---|---|---|
| Train | 841 | 1,157 | 1,059 | 816 | 852 | 963 | 5,688 |
| Validation | 169 | 222 | 224 | 215 | 176 | 179 | 1,185 |
| Test | 464 | 586 | 659 | 470 | 478 | 483 | 3,140 |

**Decisions:**
- **Validation is held out by whole groups**, not random images. A random split would put the same design in train and validation, making validation optimistic while the test is mostly unseen designs. Three groups give design diversity; validation holds 17 % of the train-plus-validation defects, with at least 169 per class.
- **No group is in both train and validation** (enforced by `src/split.py`).
- Validation groups also appear in the official test. This is not leakage: the model trains on neither. Train groups also appear in the official test by the dataset authors' design; this is reported, not changed.
- Each inspection image and its reference stay in the same row, so they can never be split apart.

## 8. Checks enforced by `src/split.py`

The script stops with an error if any of these fail:
1. Every listed inspection image, reference image and annotation file exists.
2. No group is in both train and validation.
3. No image is in more than one split.
4. All six classes are present in train and in validation.
5. Every annotation line has five integers, a non-empty box inside the image, and a class from 1 to 6.

## 9. Implications for the project

| Finding | Decision |
|---|---|
| Test mostly unseen designs | Group-level validation |
| Balanced classes | No class weights |
| No defect-free inspection images | Board false-alarm rate measured on reference images |
| Small defects (median ~35 px) | Detector with a stride-8 output grid |
| Binary, clean images | Report the gap to real camera images; recommend a client pilot |
