"""Build the GRU text classifier."""

import keras
import numpy as np


def build_model(train_texts: np.ndarray, cfg_model: dict, n_classes: int) -> keras.Model:
    """Text in, class probabilities out. Vocabulary is learned from train_texts only."""
    vectorizer = keras.layers.TextVectorization(
        max_tokens=cfg_model["vocab_size"],
        output_sequence_length=cfg_model["max_len"],
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(train_texts, batch_size=1024)

    model = keras.Sequential([
    keras.Input(shape=(), dtype="string"),
    vectorizer,
    keras.layers.Embedding(cfg_model["vocab_size"], cfg_model["embed_dim"], mask_zero=True),
    keras.layers.GRU(cfg_model["gru_units"]),
    keras.layers.Dense(n_classes, activation="softmax"),
])
    return model