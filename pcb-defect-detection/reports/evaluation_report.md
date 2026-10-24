# Evaluation report — PCB defect detection

## Targets
1. Class-agnostic AP@0.5 above the baseline by more than seed variation.
2. Class-aware mAP@0.5 on test >= 0.928 per model (secondary).
3. Board recall >= 99.5% and false-alarm rate <= 0.5% at the same threshold.

## v1 — detector only (test set, clean held-out)
- Target 1: met. AP@0.5 0.977 (both models) vs baseline 0.589 on the same test set.
- Target 2: met. mAP@0.5 0.963 (seed 0), 0.966 (seed 1).
- Target 3: not met. Recall 100% (500/500); false alarms 2.2% (11/500),
  1.81% (9/498) after excluding 2 contaminated references.
  Cause: the detector never sees the reference, so design features that look
  like defects (holes, round pads, bumps, thin necks) raise alarms.

## v2 — detector + reference check
**Target 3 — v2 on the test set (seen before, during v1):**
board recall 99.8% (499/500) >= 99.5%; false-alarm rate 0.20% (1/500, and
1/498 after excluding the 2 contaminated references) <= 0.5%. **Target 3 met.**
Because the test set was already used for v1, this is not a clean held-out result.

- The single test false alarm (90100026) is on a short isolated trace segment that
  passed the reference check (real difference vs the reference); likely an
  unlabeled open defect. Counted as a false alarm; sent for expert review.

  - The single missed board (20085311) has 2 defects: one at the left image border
  (x 8-39) that the detector did not see at all (score 0.00), and one seen at
  0.86, below the 0.91 threshold. Both pass the reference check. Known weakness:
  defects at the image border.