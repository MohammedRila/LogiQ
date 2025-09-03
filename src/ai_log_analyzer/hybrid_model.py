import warnings
from typing import Iterable, Dict, Any, Union, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ---- Portable Keras import: works with TF-Keras or Keras 3 ----
try:
    from tensorflow import keras  # TF >= 2.x
    from keras import layers, models
except Exception:  # fallback to standalone keras (e.g., keras>=3)
    import keras
    from keras import layers, models

from sentence_transformers import SentenceTransformer


class HybridLogAnalyzer:
    """
    Hybrid anomaly detection on text logs using sentence embeddings:
      - Isolation Forest (IF)
      - One-Class SVM (OCSVM)
      - Autoencoder (AE) reconstruction error
    Majority voting (>= 2 of 3) decides anomaly.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in training data (also used as nu for OCSVM).
    ae_hidden : tuple[int, int, int]
        (enc1, enc2, bottleneck) layer sizes; decoder is symmetric.
    transformer_model : str
        SentenceTransformer model name.
    epochs : int
        Autoencoder training epochs.
    batch_size : int
        Autoencoder batch size.
    random_state : int
        Reproducibility for sklearn models.
    ae_percentile : float
        Percentile for AE error threshold on training set (e.g., 95.0).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        ae_hidden: Iterable[int] = (64, 32, 16),
        transformer_model: str = "all-MiniLM-L6-v2",
        epochs: int = 20,
        batch_size: int = 32,
        random_state: int = 42,
        ae_percentile: float = 95.0,
    ):
        if not (0.0 < contamination < 0.5):
            raise ValueError("contamination should be in (0, 0.5).")
        if not (0.0 < ae_percentile < 100.0):
            raise ValueError("ae_percentile should be in (0, 100).")

        self.contamination = float(contamination)
        self.ae_hidden = tuple(int(x) for x in ae_hidden)
        if len(self.ae_hidden) != 3:
            raise ValueError("ae_hidden must be a 3-tuple like (64, 32, 16).")

        self.transformer_model_name = transformer_model
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.ae_percentile = float(ae_percentile)

        # Models / state
        self.isolation_forest: IsolationForest | None = None
        self.ocsvm: OneClassSVM | None = None
        self.autoencoder: keras.Model | None = None
        self.scaler: StandardScaler | None = None
        self.ae_threshold: float | None = None

        # Embeddings
        self.transformer = SentenceTransformer(self.transformer_model_name)

    # ------------------------------
    # Helpers
    # ------------------------------
    def _check_logs(self, logs: Iterable[str]) -> List[str]:
        if logs is None:
            raise ValueError("logs is None.")
        if not isinstance(logs, (list, tuple)):
            raise TypeError("logs must be a list/tuple of strings.")
        if len(logs) == 0:
            raise ValueError("logs must not be empty.")
        if not all(isinstance(x, str) for x in logs):
            raise TypeError("All items in logs must be strings.")
        return list(logs)

    def _build_autoencoder(self, input_dim: int) -> keras.Model:
        inp = layers.Input(shape=(input_dim,), name="ae_input")

        # Encoder
        x = layers.Dense(self.ae_hidden[0], activation="relu", name="enc_1")(inp)
        x = layers.Dense(self.ae_hidden[1], activation="relu", name="enc_2")(x)
        bottleneck = layers.Dense(self.ae_hidden[2], activation="relu", name="bottleneck")(x)

        # Decoder (symmetric)
        x = layers.Dense(self.ae_hidden[1], activation="relu", name="dec_1")(bottleneck)
        x = layers.Dense(self.ae_hidden[0], activation="relu", name="dec_2")(x)
        out = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

        model = models.Model(inputs=inp, outputs=out, name="ae")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _ensure_fitted(self):
        if any(v is None for v in [self.isolation_forest, self.ocsvm, self.autoencoder, self.scaler, self.ae_threshold]):
            raise RuntimeError("Model is not fitted. Call `fit(logs)` first.")

    # ------------------------------
    # Fit
    # ------------------------------
    def fit(self, logs: Iterable[str]) -> "HybridLogAnalyzer":
        logs = self._check_logs(logs)

        # Sentence embeddings
        embeddings = self.transformer.encode(
            logs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)

        # Scale
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(embeddings).astype(np.float32, copy=False)

        # Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X)

        # One-Class SVM
        # nu ~ expected anomaly ratio; gamma='scale' adapts to data variance
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=self.contamination)
        self.ocsvm.fit(X)

        # Autoencoder
        self.autoencoder = self._build_autoencoder(X.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.autoencoder.fit(
                X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0,
                shuffle=True,
            )

        # AE threshold from training recon error
        recon = self.autoencoder.predict(X, verbose=0)
        errors = np.mean(np.square(X - recon), axis=1)
        self.ae_threshold = float(np.percentile(errors, self.ae_percentile))

        return self

    # ------------------------------
    # Predict
    # ------------------------------
    def predict(
        self,
        logs: Iterable[str],
        return_details: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        self._ensure_fitted()
        logs = self._check_logs(logs)

        # Embeddings -> scale
        embeddings = self.transformer.encode(
            logs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        X = self.scaler.transform(embeddings).astype(np.float32, copy=False)

        # IF: -1 anomaly, 1 normal
        if_labels = (self.isolation_forest.predict(X) == -1).astype(int)
        if_scores = self.isolation_forest.decision_function(X)  # higher = more normal

        # OCSVM: -1 anomaly, 1 normal
        svm_labels = (self.ocsvm.predict(X) == -1).astype(int)
        svm_scores = self.ocsvm.decision_function(X)  # higher = more normal

        # AE reconstruction error
        recon = self.autoencoder.predict(X, verbose=0)
        ae_errors = np.mean(np.square(X - recon), axis=1)
        ae_labels = (ae_errors > self.ae_threshold).astype(int)

        # Majority vote (>= 2 of 3)
        votes = if_labels + svm_labels + ae_labels
        hybrid = np.where(votes >= 2, "Yes", "No")

        if return_details:
            return {
                "hybrid": hybrid,
                "votes": votes,
                "isolation_forest": if_labels,
                "if_score": if_scores,
                "one_class_svm": svm_labels,
                "svm_score": svm_scores,
                "autoencoder": ae_labels,
                "reconstruction_error": ae_errors,
                "ae_threshold": self.ae_threshold,
            }

        return hybrid

    # ------------------------------
    # Convenience
    # ------------------------------
    def fit_predict(
        self,
        logs: Iterable[str],
        return_details: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Fit on logs and then predict on the same logs."""
        self.fit(logs)
        return self.predict(logs, return_details=return_details)
