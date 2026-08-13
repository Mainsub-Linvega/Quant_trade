"""Minimal NumPy forward/export for sklearn-style ReLU MLP regressors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyMLP:
    def __init__(self, coefs: list[np.ndarray], intercepts: list[np.ndarray],
                 *, activation: str = "relu", output_activation: str = "identity"):
        if len(coefs) != len(intercepts) or not coefs:
            raise ValueError("coefs/intercepts must contain the same non-zero number of layers")
        if activation != "relu" or output_activation != "identity":
            raise ValueError("only relu hidden layers and identity output are supported")
        self.coefs = [np.asarray(value, dtype=np.float64) for value in coefs]
        self.intercepts = [np.asarray(value, dtype=np.float64) for value in intercepts]
        self.activation = activation
        self.output_activation = output_activation

    @classmethod
    def from_sklearn(cls, estimator: Any) -> "NumpyMLP":
        if getattr(estimator, "activation", None) != "relu":
            raise ValueError("sklearn estimator must use relu")
        if getattr(estimator, "out_activation_", None) != "identity":
            raise ValueError("sklearn estimator must use identity output")
        return cls(list(estimator.coefs_), list(estimator.intercepts_))

    def predict(self, design: np.ndarray) -> np.ndarray:
        value = np.asarray(design, dtype=np.float64)
        if value.ndim != 2:
            raise ValueError("design must be 2-dimensional")
        for index, (coef, intercept) in enumerate(zip(self.coefs, self.intercepts)):
            value = value @ coef + intercept
            if index + 1 < len(self.coefs):
                np.maximum(value, 0.0, out=value)
        return value[:, 0] if value.shape[1] == 1 else value

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "n_layers": np.array(len(self.coefs), dtype=np.int64),
            "metadata_json": np.array(json.dumps(metadata or {}, ensure_ascii=False)),
        }
        for index, (coef, intercept) in enumerate(zip(self.coefs, self.intercepts)):
            payload[f"coef_{index}"] = coef
            payload[f"intercept_{index}"] = intercept
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> tuple["NumpyMLP", dict[str, Any]]:
        with np.load(Path(path), allow_pickle=False) as payload:
            n_layers = int(payload["n_layers"])
            coefs = [payload[f"coef_{index}"] for index in range(n_layers)]
            intercepts = [payload[f"intercept_{index}"] for index in range(n_layers)]
            metadata = json.loads(str(payload["metadata_json"]))
        return cls(coefs, intercepts), metadata
