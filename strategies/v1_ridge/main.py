from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 提交包内的同目录模块。官方 runner 加载 main.py 时会把本目录压入 sys.path，
# 所以扁平 import 在本地与评测端行为一致。绝不 import 仓库的 src/ —— 提交包里没有它。
from features import apply_robust_transform, linear_predict, single_time_deviation


class Model:
    """Lightweight sequential inference model used by the official API."""

    def __init__(self):
        model_path = Path(__file__).resolve().parent / "model" / "baseline_model.json"
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.feature_columns = list(payload["selected_features"])
        self.lower = np.asarray(payload["lower"], dtype=np.float32)
        self.upper = np.asarray(payload["upper"], dtype=np.float32)
        self.center = np.asarray(payload["center"], dtype=np.float32)
        self.scale = np.asarray(payload["scale"], dtype=np.float32)
        self.intercept = np.float32(payload["intercept"])
        self.coef = np.asarray(payload["coef"], dtype=np.float32)
        self.prediction_scale = np.float32(payload["prediction_scale"])
        self.prediction_clip = np.float32(payload["prediction_clip"])
        self.last_time_id: int | None = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        raw = test.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        raw = apply_robust_transform(raw, self.lower, self.upper, self.center, self.scale)
        deviation = single_time_deviation(raw)
        return linear_predict(
            raw, deviation, self.intercept, self.coef, self.prediction_scale, self.prediction_clip
        )
