# 002 — v2: detector + reference check

Date: 2026-10-24
Status: decided, before any v2 result

## Context
v1 (detector only) met Targets 1 and 2 on the test set but not Target 3
(false alarms 1.81% vs <= 0.5%). Inspection of the 11 test false alarms:
9 are design features that look like defect types (holes like pin-hole,
round pads like copper, bumps like spur, thin necks like mousebite),
2 are contaminated reference images. The detector never sees the reference,
so it cannot tell a design feature from a defect.

## Decision
Combine the detector with a reference check:
1. Detector predicts boxes and scores on the board image.
2. Difference image: absdiff(board, reference), threshold 127,
   opening with kernel 5 (same as the baseline).
3. For each box: count white pixels of the difference image inside it.
4. Drop boxes with count < N.
5. Board score = highest score among remaining boxes (0 if none).
6. Board is defective if board score >= confidence threshold.

## How parameters are chosen (validation only)
- N and the confidence threshold are chosen on validation, with the 3
  contaminated validation references excluded.
- Safe region: all (N, threshold) pairs with recall >= 99.5% and the lowest
  false-alarm count. The chosen pair is in the middle of that region, with a
  wider margin than v1 (validation proved optimistic in v1).

## Single model vs ensemble
- Default: single model, seed 0 (fixed now, not chosen by any result).
- The 2-model ensemble (seeds 0 and 1) is used only if, on validation, at
  recall >= 99.5% it lowers the false-alarm count by at least 3 images
  compared to seed 0 alone, and total latency stays <= 50 ms on GPU.

## Fallback
If no reference exists for a board design, run v1 (detector only) and
return a warning that the false-alarm rate is higher.

## Test-set note
The test set was already used for v1. v2 test results will be reported as
"test set seen before", not as a clean held-out result.
Result on validation: both reach 0 false alarms with recall >= 99.5%
(ensemble gains 0 images, rule requires >= 3) -> single model, seed 0.
## Chosen on validation (2026-10-24)
- Preprocessing added after the stress test: median filter 3x3 on both images,
  then alignment of the board to the reference (phase correlation, never worse
  than no shift). Without it, the reference check failed at 2-3 px shift and the
  detector collapsed under pixel noise.
- Single model, seed 0 (ensemble gained 0 images; rule required >= 3).
- N = 10, threshold = 0.91: middle of the region that keeps recall >= 99.5% with
  at most 1/197 false alarm in the worst of 6 stress scenarios
  (none, shift 1/2/3 px, noise 0.2%, shift 2 px + noise).