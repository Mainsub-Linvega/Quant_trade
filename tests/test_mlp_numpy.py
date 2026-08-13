from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies" / "v4_mlp"))

from mlp_numpy import NumpyMLP


class NumpyMLPTest(unittest.TestCase):
    def test_sklearn_equivalence_and_roundtrip(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 7))
        y = np.sin(x[:, 0]) + 0.2 * x[:, 2]
        estimator = MLPRegressor(hidden_layer_sizes=(12, 5), max_iter=30, batch_size=64,
                                 random_state=5, activation="relu", solver="adam",
                                 early_stopping=False).fit(x, y)
        model = NumpyMLP.from_sklearn(estimator)
        np.testing.assert_allclose(model.predict(x), estimator.predict(x), atol=1e-12, rtol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path, {"name": "probe"})
            loaded, metadata = NumpyMLP.load(path)
            self.assertEqual(metadata, {"name": "probe"})
            np.testing.assert_allclose(loaded.predict(x), estimator.predict(x), atol=1e-12, rtol=1e-12)

    def test_rejects_non_matrix(self) -> None:
        model = NumpyMLP([np.ones((2, 1))], [np.zeros(1)])
        with self.assertRaises(ValueError):
            model.predict(np.ones(2))


if __name__ == "__main__":
    unittest.main()
