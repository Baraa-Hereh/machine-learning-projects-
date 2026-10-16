import json
import time
from pathlib import Path

import numpy as np
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess

VARIANTS = ["yolox-s", "yolox-tiny", "yolox-nano"]
SIZE = 640
NUM_CLASSES = 6
WARMUP = {"cuda": 20, "cpu": 5}
RUNS = {"cuda": 200, "cpu": 50}


def measure(model, device):
    x = torch.rand(1, 3, SIZE, SIZE, device=device)
    times = []
    with torch.no_grad():
        for i in range(WARMUP[device] + RUNS[device]):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(x)
            postprocess(out, NUM_CLASSES, 0.25, 0.45)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= WARMUP[device]:
                times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(float(np.median(times)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
    }


def main():
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    results = {"input_size": SIZE, "cpu_threads": torch.get_num_threads(), "variants": {}}
    for name in VARIANTS:
        exp = get_exp(None, name)
        exp.num_classes = NUM_CLASSES
        model = exp.get_model().eval()
        params = sum(p.numel() for p in model.parameters())
        entry = {"params_m": round(params / 1e6, 2), "size_mb": round(params * 4 / 1e6, 1)}
        for device in devices:
            entry[device] = measure(model.to(device), device)
        results["variants"][name] = entry
        print(name, entry)

    out = Path(__file__).resolve().parent.parent / "reports" / "latency.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()