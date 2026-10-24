#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

for seed in 0 1 2 3 4; do
  for exp in pcb_yolox_s_e50 pcb_yolox_s_neg_e50; do
    name="${exp}_seed${seed}"
    python third_party/YOLOX/tools/train.py -f "exps/${exp}.py" -expn "$name" -d 1 -b 8 --fp16 \
      -c third_party/YOLOX/yolox_s.pth seed "$seed" > "models/yolox/${name}.log" 2>&1
    python src/evaluate_detector.py --exp "exps/${exp}.py" --ckpt "models/yolox/${name}/best_ckpt.pth" \
      >> "models/yolox/${name}.log" 2>&1
  done
done