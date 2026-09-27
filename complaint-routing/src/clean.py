"""Clean extracted complaints and split them by time."""

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "cfpb_mask": re.compile(r"X{2,}"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}"),
    "long_digits": re.compile(r"\d{8,}"),
}


def load_config(path: Path) -> dict:
    """Read the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_pii(text: pd.Series, token: str) -> tuple[pd.Series, dict]:
    """Replace every PII pattern with one token and count matches per type."""
    counts = {}
    for name, pattern in PII_PATTERNS.items():
        counts[name] = int(text.str.count(pattern).sum())
        text = text.str.replace(pattern, token, regex=True)
    return text, counts


def remove_conflicts(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """Drop every row whose text appears with more than one label."""
    n_labels = df.groupby(key)[label].transform("nunique")
    return df[n_labels == 1]


def time_split(df: pd.DataFrame, date_col: str, train_end: str, val_end: str) -> dict:
    """Split rows by date into train, val and test."""
    d = df[date_col]
    return {
        "train": df[d <= train_end],
        "val": df[(d > train_end) & (d <= val_end)],
        "test": df[d > val_end],
    }


def clean(cfg: dict) -> dict:
    """Run all cleaning steps in order and return the splits."""
    d, c = cfg["data"], cfg["clean"]
    text_col, date_col = d["text_col"], d["date_col"]

    df = pd.read_parquet(d["extracted_path"])
    logger.info("loaded: %d", len(df))

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    logger.info("valid dates: %d", len(df))

    df = df.drop_duplicates(subset=c["id_col"])
    logger.info("unique ids: %d", len(df))

    df[text_col], pii_counts = mask_pii(df[text_col], c["mask_token"])
    logger.info("pii masked: %s", pii_counts)

    df["_key"] = df[text_col].str.strip()
    df = remove_conflicts(df, "_key", c["label_col"])
    logger.info("after conflicts: %d", len(df))

    df = df.sort_values(date_col).drop_duplicates(subset="_key", keep="first")
    df = df.drop(columns="_key")
    logger.info("after dedup: %d", len(df))

    return time_split(df, date_col, c["train_end"], c["val_end"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path("configs/config.yaml"))
    splits = clean(cfg)

    out_dir = Path(cfg["clean"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        logger.info("%s: %d rows", name, len(part))


if __name__ == "__main__":
    main()