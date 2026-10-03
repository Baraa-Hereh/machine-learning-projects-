"""Route complaints to a department with the frozen baseline model."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from baseline import load_config, run_name


def load_model(config_path: Path = Path("configs/config.yaml")):
    """Load the frozen model named by the config, plus its feature names (built once)."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    b = load_config(config_path)["baseline"]
    model = joblib.load(root / b["model_dir"] / f"{run_name(b)}.joblib")
    names = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    return model, names

def route(model, names, text: str, n_words: int = 5) -> dict:
    """Return top department, its probability, the runner-up, and the words that drove the decision."""
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]

    x = vec.transform([text])
    proba = clf.predict_proba(x)[0]
    first, second = np.argsort(proba)[::-1][:2]

    contrib = x.multiply(clf.coef_[first]).tocsr()
    order = np.argsort(contrib.data)[::-1][:n_words]
    words = names[contrib.indices[order]]

    return {
        "department": clf.classes_[first],
        "probability": round(float(proba[first]), 3),
        "runner_up": clf.classes_[second],
        "runner_up_probability": round(float(proba[second]), 3),
        "top_words": [w for w, v in zip(words, contrib.data[order]) if v > 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="complaint text")
    args = parser.parse_args()

    model, names = load_model()
    start = time.perf_counter()
    result = route(model, names, args.text)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()