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

**Target 3 — v2 on the test set (seen before, during v1), kernel 3, N 10, threshold 0.94:**
board recall 99.8% (499/500) >= 99.5%; false-alarm rate 0.40% (2/500, and 2/498
after excluding the 2 contaminated references) <= 0.5%. **Target 3 met.**
Not a clean held-out result: the test set was used for v1.

- False alarms: 90100026 (short isolated trace segment that passed the reference
  check; likely an unlabeled open, sent for expert review) and 12000068 (round
  design hole; small edge residue survives the kernel-3 opening).
- Missed board: 20085311 (one defect at the image border not detected; one seen
  at 0.86, below threshold).
- Worst case in the 6-scenario stress test at threshold 0.94: false alarms 3/197 (1.52%).
- Previous setting (kernel 5) gave 1/500 false alarms on test but erased 111 of
  1185 validation defects from the difference image; replaced on 2026-11-03.