import shutil
from pathlib import Path

import pandas as pd

from baseline import load_config

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg, root = load_config(ROOT / "configs/config.yaml")
    raw = root / cfg["data"]["raw_dir"]
    out = ROOT / "references"
    out.mkdir(exist_ok=True)
    for split in ["val", "test"]:
        df = pd.read_csv(root / cfg["data"]["splits_dir"] / f"{split}.csv")
        for p in df["ref_img"]:
            shutil.copy(raw / p, out / f"{Path(p).stem.replace('_temp', '')}.jpg")
    print(f"{len(list(out.glob('*.jpg')))} references in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()