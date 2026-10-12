# Project Brief — PCB Defect Detection

**Status:** agreed scope, before data audit
**Date:** 10 October 2026

## 1. Client and problem

A manufacturer of printed circuit boards inspects bare boards after the copper pattern is printed and before components are mounted.

**Current process.** An automated optical inspection (AOI) machine compares each board image with a defect-free reference using fixed rules. It raises many false alarms (dust, lighting changes, small print variations), so an operator re-inspects every flagged board. Operators tire and miss defects over a shift.

**Why it matters.**
- A **missed defect** gets components mounted on it. Each later stage multiplies the cost; a defect found by the end customer is far more expensive than one caught here.
- A **false alarm** costs operator time and slows the line.

**What success means.** Catch almost all defective boards, with fewer false alarms than the current machine, fast enough to keep up with the line. Zero errors is not achievable; missed defects are the costlier error.

## 2. Task

Given a 640×640 grayscale image of a board region, return every defect as a box with its type and a confidence score. A board region with no defects returns an empty list.

Defect types (6): open, short, mousebite, spur, spurious copper, pin-hole.

## 3. Metrics

| Metric | Level | Purpose |
|---|---|---|
| **mAP@0.5** (main) | defect | Model quality across all confidence thresholds; comparable with published DeepPCB results |
| mAP@0.5:0.95 | defect | Localisation precision |
| Board recall | board | Share of defective boards that are stopped — the client's actual decision |
| Board false-alarm rate | board | Share of defect-free boards stopped by mistake — operator workload; measured on the defect-free reference images |
| Per-class AP | defect | No defect type hidden behind the common ones |

Recall is the priority, with a ceiling on false alarms; otherwise the operator workload returns.

## 4. Baseline and target

**Baseline:** reference-image comparison, the same principle as the client's current AOI. Subtract the defect-free reference from the board image, threshold the difference, box each region of difference. No learning, no defect types.

**Target:** set after the baseline is measured, in the same form as the main comparison:
- higher board recall than the baseline at an equal or lower board false-alarm rate, and
- a mAP@0.5 target agreed once the baseline and a first model are measured.
**Update (12 October 2026):** baseline measured; targets set in `reports/baseline.md`, section 6.
## 5. Constraints

| Constraint | Value | Reason |
|---|---|---|
| Latency | ≤ 50 ms per 640×640 region on the line computer | A full board is many regions; the total must keep up with the line |
| Hardware | Local industrial PC by the line (CPU or small GPU), no cloud | Runs on site |
| Model size | ≤ 100 MB | Limited device, easy updates |
| Data privacy | Images never leave the plant | Board designs are trade secrets |
| Fail-safe | If the system fails or cannot decide, the board goes to manual inspection | Slowing down is cheaper than passing a defective board |
| Explanation | Every decision shows boxes on the defects | The operator must see where to repair |
| Licences | All libraries must allow commercial use without forcing the client to open its code | Some popular detection libraries are licensed under terms that require this or a paid licence |

## 6. Data

**DeepPCB** (MIT licence): 1,500 pairs of 640×640 grayscale images — a board region with defects and its defect-free reference — with one annotation file per pair (`x1 y1 x2 y2 class` per defect). Official split files `trainval.txt` and `test.txt` are provided.

Known so far:
- 1,501 reference images for 1,500 test images; one reference (`90100034_temp.jpg`) has no pair and is excluded.
- The split files list images without the `_test` / `_temp` suffix.
- Defects are small (around 30 px in a 640 px image), so the model needs a fine detection grid.

To be checked in the data audit: class balance, defects per image, box sizes, alignment between each pair, duplicates and leakage between the official splits.

**Split rule:** a board image and its reference always stay in the same split.

## 7. Approach (outline)

1. Data audit and report.
2. Baseline (reference comparison) and its scores.
3. Detection model starting from pretrained weights (small dataset), with one change at a time and five seeds.
4. Error analysis on validation; one-time evaluation on the test set.
5. Deployment as a local inference service.

## 8. Deliverables

- Trained model and inference service (local, containerised).
- Reproducible training and evaluation scripts.
- Client README, evaluation report, and project delivery notes.

## 9. Out of scope

- Integration with the production line, PLC, camera or lighting.
- Assembled boards (component placement defects).
- Automatic periodic retraining.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Small dataset (1,500 images) | Pretrained weights, augmentation, careful validation |
| Very small defects | Detector with a fine (stride 8) output grid |
| DeepPCB images are cleaner and better aligned than a real line camera | State the gap in the report; recommend a pilot on client images |
| Tool licences | Check every library's licence before adopting it |

## 11. Open questions for the client

1. Actual line speed (boards per minute, regions per board)?
2. Actual line computer (CPU / GPU)?
3. Acceptable board false-alarm rate?
4. Are defective boards repaired or scrapped? (affects how much defect-level recall matters)
