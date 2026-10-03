"""Evaluate the frozen baseline model on the test split. Runs once per model."""

import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import classification_report

from baseline import evaluate, load_config, load_split, run_name

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path("configs/config.yaml"))
    d, c, b = cfg["data"], cfg["clean"], cfg["baseline"]

    name = run_name(b)
    model_path = Path(b["model_dir"]) / f"{name}.joblib"
    report = Path(b["report_dir"]) / f"{name}_test.json"
    if report.exists():
        raise FileExistsError(f"{report} already exists: the test set is evaluated once per model")
    logger.info("model: %s | report: %s", model_path, report)

    model = joblib.load(model_path)
    X_te, y_te = load_split(c["out_dir"], "test", d["text_col"], c["label_col"])

    result = evaluate("tfidf_lr test", model, X_te, y_te)
    print(classification_report(y_te, model.predict(X_te), zero_division=0))

    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("saved test report")


if __name__ == "__main__":
    main()