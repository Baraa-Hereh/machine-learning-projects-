"""Train and evaluate the majority and TF-IDF + logistic regression baselines on validation."""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Read the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(split_dir: str, name: str, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    """Load one split and return texts and labels."""
    df = pd.read_parquet(Path(split_dir) / f"{name}.parquet")
    return df[text_col], df[label_col]


def evaluate(name: str, model, X: pd.Series, y: pd.Series) -> dict:
    """Score a fitted model on one split."""
    pred = model.predict(X)
    macro = f1_score(y, pred, average="macro")
    weighted = f1_score(y, pred, average="weighted")
    logger.info("%s | macro F1 %.3f | weighted F1 %.3f", name, macro, weighted)
    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path("configs/config.yaml"))
    d, c, b = cfg["data"], cfg["clean"], cfg["baseline"]

    X_tr, y_tr = load_split(c["out_dir"], "train", d["text_col"], c["label_col"])
    X_va, y_va = load_split(c["out_dir"], "val", d["text_col"], c["label_col"])

    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)

    tfidf_lr = make_pipeline(
        TfidfVectorizer(max_features=b["max_features"], ngram_range=tuple(b["ngram_range"]),
                        min_df=b["min_df"], sublinear_tf=True),
        LogisticRegression(C=b["C"], max_iter=b["max_iter"], class_weight="balanced"),
    ).fit(X_tr, y_tr)

    results = {
        "majority": evaluate("majority", majority, X_va, y_va),
        "tfidf_lr": evaluate("tfidf_lr", tfidf_lr, X_va, y_va),
    }

    print(classification_report(y_va, tfidf_lr.predict(X_va), zero_division=0))

    Path(b["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(b["report_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    Path(b["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf_lr, b["model_path"])
    logger.info("saved report and model")


if __name__ == "__main__":
    main()