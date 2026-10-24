import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED = {"train": (800, 5688), "val": (200, 1185), "test": (500, 3140)}


def convert(df, raw_dir, classes, size, with_refs=False):
    images, annotations = [], []
    ann_id = 1
    for image_id, row in enumerate(df.itertuples(), start=1):
        images.append({"id": image_id, "file_name": row.test_img, "width": size, "height": size})
        for x1, y1, x2, y2, cls in np.loadtxt(raw_dir / row.ann, dtype=int, ndmin=2):
            w, h = int(x2 - x1), int(y2 - y1)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [int(x1), int(y1), w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
    if with_refs:
        start = len(images) + 1
        for image_id, row in enumerate(df.itertuples(), start=start):
            images.append({"id": image_id, "file_name": row.ref_img, "width": size, "height": size})
    categories = [{"id": i, "name": name} for i, name in enumerate(classes, start=1)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    root = Path(__file__).resolve().parent.parent
    with open(root / "configs/config.yaml") as f:
        cfg = yaml.safe_load(f)["data"]
    raw_dir = root / cfg["raw_dir"]
    out_dir = root / cfg["coco_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, (n_img, n_ann) in EXPECTED.items():
        df = pd.read_csv(root / cfg["splits_dir"] / f"{split}.csv")
        coco = convert(df, raw_dir, cfg["classes"], cfg["image_size"])
        got = (len(coco["images"]), len(coco["annotations"]))
        if got != (n_img, n_ann):
            raise ValueError(f"{split}: expected {n_img} images / {n_ann} boxes, got {got[0]} / {got[1]}")
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(coco, f)
        print(f"{split}: {got[0]} images, {got[1]} boxes -> {out_dir / f'{split}.json'}")
        df = pd.read_csv(root / cfg["splits_dir"] / "train.csv")
    coco = convert(df, raw_dir, cfg["classes"], cfg["image_size"], with_refs=True)
    got = (len(coco["images"]), len(coco["annotations"]))
    if got != (1600, 5688):
        raise ValueError(f"train_with_refs: expected 1600 images / 5688 boxes, got {got[0]} / {got[1]}")
    with open(out_dir / "train_with_refs.json", "w") as f:
        json.dump(coco, f)
    print(f"train_with_refs: {got[0]} images, {got[1]} boxes")

if __name__ == "__main__":
    main()