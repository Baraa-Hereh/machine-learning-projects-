"""Extract complaints with narratives for a date range from the raw CFPB parquet shards."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Read the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_shard(path: Path, d: dict) -> pd.DataFrame:
    """Load one shard and keep rows in the date range that have text."""
    df = pd.read_parquet(path, columns=d["columns"])
    dates = pd.to_datetime(df[d["date_col"]], errors="coerce")
    in_range = dates.between(d["start_date"], d["end_date"])
    has_text = df[d["text_col"]].fillna("").str.strip().ne("")
    return df[in_range & has_text]


def extract(cfg: dict) -> pd.DataFrame:
    """Filter every shard in the raw folder and combine the results."""
    d = cfg["data"]
    shards = sorted(Path(d["raw_dir"]).glob(d["raw_glob"]))
    if not shards:
        raise FileNotFoundError(f"no files matching {d['raw_glob']} in {d['raw_dir']}")

    kept = []
    for path in shards:
        part = filter_shard(path, d)
        kept.append(part)
        logger.info("%s: kept %d rows", path.name, len(part))

    return pd.concat(kept, ignore_index=True)


def main() -> None:
    cfg = load_config(Path("configs/config.yaml"))
    df = extract(cfg)
    out = Path(cfg["data"]["extracted_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("saved %d rows from %d columns to %s", len(df), df.shape[1], out)


if __name__ == "__main__":
    main()