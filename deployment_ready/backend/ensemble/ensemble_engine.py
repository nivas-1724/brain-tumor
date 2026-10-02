"""
Ensemble Prediction Engine
Combines predictions from top-performing models using Weighted Probability Averaging and Simple Probability Averaging.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class EnsemblePredictor:
    def __init__(self, models_dict=None, weights_dict=None):
        """
        models_dict: dictionary of name -> loaded Keras model
        weights_dict: dictionary of name -> float weight (e.g. F1 score)
        """
        self.models = models_dict or {}
        self.weights = weights_dict or {}

    def predict_probs(self, X, method="weighted"):
        """
        Returns ensemble probability predictions for input batch X (N, 224, 224, 3).
        Methods:
          - "weighted": Weighted probability averaging
          - "average": Equal probability averaging
          - "majority": Majority vote based on argmax
        """
        if not self.models:
            raise ValueError("No models loaded in EnsemblePredictor.")

        model_names = list(self.models.keys())
        predictions = []
        for name in model_names:
            m = self.models[name]
            try:
                preds = m(X, training=False).numpy()
            except Exception:
                preds = m.predict(X, verbose=0)
            predictions.append(preds)

        if method == "average":
            return np.mean(predictions, axis=0)

        elif method == "majority":
            # Hard voting
            votes = np.array([np.argmax(p, axis=1) for p in predictions])  # (M, N)
            num_classes = predictions[0].shape[1]
            N = X.shape[0]
            prob_out = np.zeros((N, num_classes), dtype=np.float32)
            for i in range(N):
                sample_votes = votes[:, i]
                counts = np.bincount(sample_votes, minlength=num_classes)
                prob_out[i] = counts / len(model_names)
            return prob_out

        else:
            # Weighted probability averaging (default)
            w_list = [self.weights.get(name, 1.0) for name in model_names]
            total_w = sum(w_list)
            norm_weights = [w / total_w for w in w_list]

            weighted_probs = np.zeros_like(predictions[0])
            for p, w in zip(predictions, norm_weights):
                weighted_probs += w * p
            return weighted_probs

