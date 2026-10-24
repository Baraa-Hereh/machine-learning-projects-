# Baseline — Reference Comparison

**Date:** 12 October 2026
**Script:** `src/baseline.py` · **Exploration:** `notebooks/02_baseline.ipynb`
**Split:** validation (groups 44000, 50600, 77000; 200 images, 1,185 defects)

## 1. Method

No learning, no defect types. For each board image and its reference:

1. Absolute difference, thresholded at 127 (JPEG leaves some grey pixels).
2. Morphological opening (erosion then dilation) with a square kernel, to remove thin edge lines caused by misalignment.
3. Connected components (8-connectivity); one box per component.
4. Each box is enlarged by a fixed margin on every side, because annotation boxes are drawn with space around the defect while component boxes fit the pixels tightly.
5. Confidence score = component area in pixels (larger regions are more likely to be real defects than leftover edge fragments).

Evaluation is class-agnostic, since the baseline gives no defect type. Boxes are matched to annotations within each image (IoU, one prediction per annotation, highest score first); precision-recall is computed over all images together.

## 2. Parameter selection (validation, AP@0.5)

| Kernel | Margin 9 | 12 | 15 | 18 | 21 |
|---|---|---|---|---|---|
| 3 | 0.526 | 0.584 | 0.596 | 0.290 | 0.053 |
| 5 | 0.533 | 0.663 | **0.698** | 0.458 | 0.130 |
| 7 | 0.226 | 0.303 | 0.332 | 0.279 | 0.109 |

- Without a margin, AP@0.5 is near zero (0.002): the baseline finds the defects, but its boxes are smaller than the annotations.
- Beyond 15 px, boxes become larger than the annotations and AP falls.
- Kernel 7 removes parts of real defects.

**Selected:** kernel 5, margin 15. The full grid is in `reports/baseline_val_grid.csv`.

## 3. Results (validation)

| Metric | Value |
|---|---|
| AP@0.5 (class-agnostic) | **0.698** |
| AP@0.5:0.95 (class-agnostic) | 0.232 |
| Per-class AP | not available (no defect types) |

Margin 15 maximises AP@0.5 at the cost of localisation: with margin 12, AP@0.5:0.95 was 0.328.

**Board level.** A board is stopped if any box has an area at or above the threshold.

| Area threshold (px) | Board recall | Board false-alarm rate |
|---|---|---|
| 1 | 99.5% | 0.5% |
| 10 | 99.5% | 0.5% |
| 25 | 99.5% | 0.0% |
| 50 | 95.5% | 0.0% |
| 100 | 87.5% | 0.0% |
| 200 | 58.0% | 0.0% |
| 400 | 18.0% | 0.0% |

One defective board (of 200) produced no box at any threshold: the opening removed all its defects.

## 4. How the false-alarm rate was measured

Every inspection image contains defects, so defect-free boards were simulated from the reference images. Comparing a reference with itself always gives zero difference, so each reference was shifted to mimic a new photograph of the same board.

Shifts were measured on the 800 training pairs (the shift that minimises the difference):

| Shift (px) | Pairs |
|---|---|
| 0 | 79.1% |
| 1 | 13.5% |
| 2–3 | 3.5% |
| 4–8 | 3.9% (piles up at the search limit; treated as failed measurements) |

Each validation reference received a shift drawn from the 0–3 px distribution (seed 42).

**Limitation:** this simulates misalignment only, not dust, lighting or camera noise. On a real line the false-alarm rate will be higher.

## 5. Findings

- On DeepPCB the baseline is almost perfect at board level. The images are binary, clean and mostly aligned; a real camera will be harder.
- Box quality is the weak point: AP@0.5 0.698 needs a tuned margin, and localisation is poor (AP@0.5:0.95 0.232).
- The baseline cannot tell defect types apart.

## 6. Targets for the detection model

1. **Class-agnostic AP@0.5** above 0.698, by more than the seed-to-seed variation (measured on the first model with five seeds).
2. **Target 2 (secondary): class-aware mAP@0.5 on the test set**

- Target: class-aware mAP@0.5 on the test set >= **0.928**, for each of the two
  ensemble models (seeds 0 and 1), measured with the official YOLOX COCO evaluator.
- Derivation: validation mean over 5 seeds (0.9396) minus 2 x standard deviation (0.0059).
  Per-seed validation values: 0.935, 0.940, 0.933, 0.948, 0.942.
- Purpose: a non-regression check. It confirms that defect-type recognition does not
  collapse on unseen boards. The client needs the defect type only approximately;
  the primary goal is detecting defective boards (Target 3).
- Why per model: the ensemble averages board-level scores only, not class predictions,
  so class-aware mAP is measured on each model separately.
  **Test-set reference image procedure (fixed before running the test evaluation)**

1. Every reference image with ensemble board score >= 0.81 is inspected.
2. Each one is compared with its test image at the same location, using fixed
   design features as anchors, and checked against the test label file.
3. Classification:
   - Case 1: feature present in both images and not labeled -> kept as a false alarm,
     sent for expert review.
   - Case 2: feature present in the reference only -> reference is contaminated,
     excluded from the false-alarm count.
   - Case 3: feature present in both images and labeled as a defect in the test image
     -> reference is contaminated, excluded from the false-alarm count.
4. Both numbers are reported: false-alarm rate on all reference images, and on the
   reference images left after exclusions, with every excluded image listed by name.
5. No other changes are made after seeing test results (threshold, models, seeds,
   score mode stay as fixed in configs/config.yaml).
3. **Board level:** recall ≥ 99.5% at a false-alarm rate ≤ 0.5%, i.e. no worse than the baseline. A higher board recall cannot be shown on this data: at most one board of 200 is left to improve.