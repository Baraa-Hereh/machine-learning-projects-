# Decision 001 — Detection model

**Date:** 16 October 2026
**Status:** model family decided; size variant pending latency measurement

## Context

The detector must meet the constraints in `docs/project_brief.md`, section 5:
- licence that allows commercial use without forcing the client to open its code,
- model size ≤ 100 MB,
- latency ≤ 50 ms per 640×640 region on the line computer,
- a fine output grid (stride 8) for small defects (median ~35 px),
- pretrained weights, since the training set is small (800 images).

## Options considered

| Option | Licence | Size (approx.) | Outcome |
|---|---|---|---|
| Ultralytics YOLO (v8, 11, 26) | AGPL-3.0 (or paid enterprise licence) | small | Rejected: licence |
| YOLOv6, v7, v9 | GPL-3.0 | small | Rejected: licence |
| RetinaNet / FCOS (torchvision, ResNet-50) | BSD-3-Clause | ~130 MB | Rejected: size |
| RT-DETR-R18 | Apache-2.0 | ~80 MB | Not chosen: more complex and slower to train; no reason to pay that cost before trying a simpler model |
| **YOLOX** | **Apache-2.0** | **~35 MB (S)** | **Chosen** |

## Decision
- Code version: `Megvii-BaseDetection/YOLOX` at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (package version 0.3.0), installed in `third_party/YOLOX`.
Use **YOLOX**, starting from COCO-pretrained weights, fine-tuned on DeepPCB.

- Licence verified from the official repository (`Megvii-BaseDetection/YOLOX`, `LICENSE`, Apache-2.0, Copyright 2021–2022 Megvii Inc.) on 15 October 2026. No `NOTICE` file.
- Output grid includes stride 8.
- Anchor-free design with IoU-based box loss, consistent with the approach reviewed for this project.

## Adaptation to the data

- Number of classes: 80 → 6 (new classification layer, trained from scratch; all other layers fine-tuned from pretrained weights).
- Grayscale images are repeated into 3 channels, so the pretrained first layer is kept.
- Annotation files are converted to COCO format using the existing splits (`data/processed/splits/`). Image content, boxes and splits are unchanged, so results stay comparable with the baseline.

## Pending
- Latency measured on the development laptop (`reports/latency.json`): all variants meet 50 ms on GPU (S: p95 16.8 ms); none meets it on CPU (S: p95 186 ms, Nano: p95 69 ms). Training continues with YOLOX-S. The final variant depends on the client's line computer (open question 2) and on CPU optimisation tested at deployment (ONNX runtime, reduced precision).
- **Size variant (S, Tiny or Nano):** chosen after measuring latency on the target hardware (GPU and CPU). Start with YOLOX-S; move to a smaller variant if it misses 50 ms.

## Risks

| Risk | Mitigation |
|---|---|
| Latency on a CPU-only line computer | Measure early; fall back to Tiny or Nano |
| Repository last heavily maintained in 2022–2023; possible incompatibility with recent PyTorch | Pin versions; fix and document any patch (Apache-2.0 §4(b): mark modified files) |

## Licence obligations (Apache-2.0)

Ship a copy of the licence, keep copyright notices in YOLOX files, mark modified files, and do not use the YOLOX or Megvii names as product branding. Tracked in `THIRD_PARTY_LICENSES.md`.