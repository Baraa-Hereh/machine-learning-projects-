"""Train the text classifier with several seeds and early stopping on validation Macro F1."""

import json
import logging
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from model import build_model

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Read the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(split_dir: str, name: str, text_col: str, label_col: str,
               labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load one split; fail if it contains a label not in the config list."""
    df = pd.read_parquet(Path(split_dir) / f"{name}.parquet")
    unknown = set(df[label_col]) - set(labels)
    if unknown:
        raise ValueError(f"unknown labels in {name}: {unknown}")
    y = df[label_col].map({label: i for i, label in enumerate(labels)}).to_numpy()
    return df[text_col].to_numpy(), y


class MacroF1EarlyStopping(keras.callbacks.Callback):
    """Score Macro F1 on validation and on a fixed train sample each epoch; stop on validation only."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray,
                 X_trs: np.ndarray, y_trs: np.ndarray, patience: int):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.X_trs, self.y_trs = X_trs, y_trs
        self.patience = patience
        self.best, self.best_epoch, self.best_weights, self.wait = -1.0, 0, None, 0
        self.history, self.train_history = [], []

    def _macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.model.predict(X, batch_size=512, verbose=0).argmax(axis=1)
        return f1_score(y, pred, average="macro")

    def on_epoch_end(self, epoch, logs=None):
        score = self._macro_f1(self.X_val, self.y_val)
        train_score = self._macro_f1(self.X_trs, self.y_trs)
        self.history.append(score)
        self.train_history.append(train_score)
        logger.info("epoch %d | train-sample macro F1 %.4f | val macro F1 %.4f | gap %.4f",
                    epoch + 1, train_score, score, train_score - score)
        if score > self.best:
            self.best, self.best_epoch, self.wait = score, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
   
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

def run_name(cfg_model: dict) -> str:
        """Name for model and report paths, e.g. 'mean' or 'mean_sdrop30'."""
        name = cfg_model["encoder"]
        rate = cfg_model["embed_dropout"]
        if rate > 0:
            name += f"_sdrop{round(rate * 100)}"
        return name
def train_one(seed: int, X_tr, y_tr, X_va, y_va, X_trs, y_trs, cfg: dict, n_classes: int,
              class_weight: dict) -> dict:
    """Train one model with one seed and save it."""
    t = cfg["train"]
    keras.utils.set_random_seed(seed)
    model = build_model(X_tr, cfg["model"], n_classes)
    model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]),
                  loss="sparse_categorical_crossentropy")

    stopper = MacroF1EarlyStopping(X_va, y_va, X_trs, y_trs, t["patience"])
    model.fit(X_tr, y_tr, batch_size=t["batch_size"], epochs=t["max_epochs"],
              class_weight=class_weight, callbacks=[stopper], verbose=2)

    name = run_name(cfg["model"])
    out = Path(t["out_dir"]) / name / f"{name}_seed{seed}.keras"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return {"seed": seed, "best_macro_f1": stopper.best, "best_epoch": stopper.best_epoch,
            "epochs_run": len(stopper.history), "history": stopper.history,
            "train_history": stopper.train_history}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path("configs/config.yaml"))
    name = run_name(cfg["model"])
    report = Path(cfg["train"]["report_dir"]) / f"{name}_val.json"
    logger.info("run: %s | report: %s", name, report)

    d, c, t, labels = cfg["data"], cfg["clean"], cfg["train"], cfg["labels"]

    X_tr, y_tr = load_split(c["out_dir"], "train", d["text_col"], c["label_col"], labels)
    X_va, y_va = load_split(c["out_dir"], "val", d["text_col"], c["label_col"], labels)

    weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_tr)
    class_weight = dict(enumerate(weights))
    logger.info("class weights: %s", {labels[i][:20]: round(w, 2) for i, w in class_weight.items()})

    rng = np.random.default_rng(t["train_eval_seed"])
    idx = rng.choice(len(X_tr), size=min(t["train_eval_size"], len(X_tr)), replace=False)
    X_trs, y_trs = X_tr[idx], y_tr[idx]
    logger.info("train eval sample: %d of %d", len(idx), len(X_tr))

    results = [train_one(s, X_tr, y_tr, X_va, y_va, X_trs, y_trs, cfg, len(labels), class_weight)
               for s in t["seeds"]]

    scores = [r["best_macro_f1"] for r in results]
    logger.info("macro F1 mean %.4f | min %.4f | max %.4f",
                np.mean(scores), min(scores), max(scores))

    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()