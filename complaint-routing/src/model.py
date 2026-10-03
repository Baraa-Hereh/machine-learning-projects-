"""Build the text classifier: vectorizer, embedding, encoder, softmax output."""

import keras
import numpy as np
import tensorflow as tf
@keras.saving.register_keras_serializable(package="complaint_routing")
class AttentionPooling(keras.layers.Layer):
    """Learned weighted average over time steps; padded steps get zero weight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.score = keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        scores = keras.ops.squeeze(self.score(inputs), axis=-1)
        if mask is not None:
            scores = keras.ops.where(mask, scores, -1e9)
        weights = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * keras.ops.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None
@keras.saving.register_keras_serializable(package="complaint_routing")
def split_with_bigrams(text):
    """Split text into words, then add adjacent word pairs joined by '_'."""
    words = tf.strings.split(text)
    return tf.strings.ngrams(words, ngram_width=[1, 2], separator="_")
ENCODERS = {
    "gru": lambda cfg: keras.layers.GRU(cfg["gru_units"]),
    "mean": lambda cfg: keras.layers.GlobalAveragePooling1D(),
        "attention": lambda cfg: AttentionPooling(),
}


def build_model(train_texts: np.ndarray, cfg_model: dict, n_classes: int) -> keras.Model:
    """Text in, class probabilities out. Vocabulary is learned from train_texts only."""
    encoder = cfg_model["encoder"]
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder '{encoder}', expected one of {list(ENCODERS)}")
    rate = cfg_model["embed_dropout"]
    ngrams = cfg_model["ngrams"]
    if ngrams not in (1, 2):
        raise ValueError(f"ngrams must be 1 or 2, got {ngrams}")
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"embed_dropout must be in [0, 1), got {rate}")

    vectorizer = keras.layers.TextVectorization(
        max_tokens=cfg_model["vocab_size"],
        output_sequence_length=cfg_model["max_len"],
        standardize="lower_and_strip_punctuation",
        split=split_with_bigrams if ngrams == 2 else "whitespace",
    )
    vectorizer.adapt(train_texts, batch_size=1024)

    layers = [
        keras.Input(shape=(), dtype="string"),
        vectorizer,
        keras.layers.Embedding(cfg_model["vocab_size"], cfg_model["embed_dim"], mask_zero=True),
    ]
    if rate > 0:
        layers.append(keras.layers.SpatialDropout1D(rate))
    layers += [ENCODERS[encoder](cfg_model), keras.layers.Dense(n_classes, activation="softmax")]
    return keras.Sequential(layers)