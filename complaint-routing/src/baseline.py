"""Train and evaluate the majority and TF-IDF + logistic regression baselines on validation."""

import json
import logging
from pathlib import Path
import numpy as np 
import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

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
def run_name(b: dict) -> str:
    """Name for report and model paths, e.g. 'tfidf_50k_ng12' or 'tfidf_20k_ng11'."""
    lo, hi = b["ngram_range"]
    name = f"tfidf_{b['max_features'] // 1000}k_ng{lo}{hi}"
    if b["class_weight_power"] != 1.0:
        name += f"_cw{round(b['class_weight_power'] * 100)}"
    name += f"_C{b['C']:g}"
    return name

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path("configs/config.yaml"))
    d, c, b = cfg["data"], cfg["clean"], cfg["baseline"]
    name = run_name(b)
    report = Path(b["report_dir"]) / f"{name}_val.json"
    model_path = Path(b["model_dir"]) / f"{name}.joblib"
    logger.info("run: %s | report: %s", name, report)
    X_tr, y_tr = load_split(c["out_dir"], "train", d["text_col"], c["label_col"])
    X_va, y_va = load_split(c["out_dir"], "val", d["text_col"], c["label_col"])

    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    classes = np.unique(y_tr)
    balanced = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {k: w ** b["class_weight_power"] for k, w in zip(classes, balanced)}
    logger.info("class weights: %s", {k[:20]: round(w, 2) for k, w in class_weight.items()})
    tfidf_lr = make_pipeline(
        TfidfVectorizer(max_features=b["max_features"], ngram_range=tuple(b["ngram_range"]),
                        min_df=b["min_df"], sublinear_tf=True),
        LogisticRegression(C=b["C"], max_iter=b["max_iter"], class_weight=class_weight),
    ).fit(X_tr, y_tr)

    t = cfg["train"]
    rng = np.random.default_rng(t["train_eval_seed"])
    idx = rng.choice(len(X_tr), size=min(t["train_eval_size"], len(X_tr)), replace=False)

    results = {
        "majority": evaluate("majority", majority, X_va, y_va),
        "tfidf_lr": evaluate("tfidf_lr", tfidf_lr, X_va, y_va),
        "tfidf_lr_train_sample": evaluate("tfidf_lr train-sample", tfidf_lr,
                                          X_tr.iloc[idx], y_tr.iloc[idx]),
    }
    gap = results["tfidf_lr_train_sample"]["macro_f1"] - results["tfidf_lr"]["macro_f1"]
    logger.info("tfidf_lr gap (train-sample - val) %.3f", gap)
    print(classification_report(y_va, tfidf_lr.predict(X_va), zero_division=0))

    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf_lr, model_path)
    logger.info("saved report and model")

if __name__ == "__main__":
    main()