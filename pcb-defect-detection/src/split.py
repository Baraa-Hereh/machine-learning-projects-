"""Build train / val / test split files from the DeepPCB official lists, with group-level validation."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_official(raw: Path, name: str) -> pd.DataFrame:
    """One row per image pair in an official list; adds the _test / _temp suffixes."""
    rows = []
    for line in (raw / f"{name}.txt").read_text().splitlines():
        if not line.strip():
            continue
        img, ann = line.split()
        stem = img[:-4]
        rows.append({"id": Path(stem).name, "group": Path(stem).parts[0],
                     "test_img": f"{stem}_test.jpg", "ref_img": f"{stem}_temp.jpg", "ann": ann})
    return pd.DataFrame(rows)


def check_annotations(df: pd.DataFrame, raw: Path, size: int, n_classes: int) -> set[int]:
    """Fail on any malformed annotation line; return the classes present."""
    present = set()
    for r in df.itertuples():
        for i, line in enumerate((raw / r.ann).read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{r.ann}:{i} expected 5 values, got {len(parts)}")
            x1, y1, x2, y2, c = map(int, parts)
            if not (0 <= x1 < x2 <= size and 0 <= y1 < y2 <= size):
                raise ValueError(f"{r.ann}:{i} box outside image or empty: {parts}")
            if not 1 <= c <= n_classes:
                raise ValueError(f"{r.ann}:{i} class {c} out of range")
            present.add(c)
    return present


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = load_config(Path("configs/config.yaml"))["data"]
    raw, out = Path(d["raw_dir"]), Path(d["splits_dir"])

    trainval = read_official(raw, "trainval")
    test = read_official(raw, "test")
    is_val = trainval["group"].isin(d["val_groups"])
    splits = {"train": trainval[~is_val], "val": trainval[is_val], "test": test}

    for name, df in splits.items():
        for col in ["test_img", "ref_img", "ann"]:
            missing = [p for p in df[col] if not (raw / p).exists()]
            if missing:
                raise FileNotFoundError(f"{name}: {len(missing)} missing {col}, e.g. {missing[0]}")

    shared = set(splits["train"]["group"]) & set(splits["val"]["group"])
    if shared:
        raise ValueError(f"groups in both train and val: {sorted(shared)}")

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared_ids = set(splits[a]["id"]) & set(splits[b]["id"])
        if shared_ids:
            raise ValueError(f"images in both {a} and {b}: {len(shared_ids)}")

    n_classes = len(d["classes"])
    for name in ["train", "val"]:
        present = check_annotations(splits[name], raw, d["image_size"], n_classes)
        absent = set(range(1, n_classes + 1)) - present
        if absent:
            raise ValueError(f"{name} has no examples of classes {sorted(absent)}")
    check_annotations(splits["test"], raw, d["image_size"], n_classes)

    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
        logger.info("%s: %d images, %d groups", name, len(df), df["group"].nunique())


if __name__ == "__main__":
    main()