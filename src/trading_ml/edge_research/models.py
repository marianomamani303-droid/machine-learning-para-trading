from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trading_ml.common import read_json, write_new_json
from trading_ml.edge_research.features import EXTRA, PRICE


def matrix(frame, recipe):
    x = frame[list(PRICE)].to_numpy(dtype=float)
    if recipe == "logistic_noise":
        # Deterministic row-wise control: hash time, never targets or future data.
        keys = (frame.timestamp.dt.as_unit("ns").astype("int64") // 1000000).to_numpy(
            dtype=np.uint64
        )
        keys ^= np.uint64(20261017)
        noise = []
        for k in range(len(EXTRA)):
            z = keys + np.uint64((k + 1) * 987654321)
            z ^= z >> np.uint64(16)
            z *= np.uint64(0x45D9F3B)
            z ^= z >> np.uint64(16)
            u = ((z % np.uint64(10000019)).astype(float) + 0.5) / 10000019
            noise.append(ndtri(u))
        x = np.column_stack([x, *noise])
    elif recipe != "logistic_price":
        x = np.column_stack([x, frame[list(EXTRA)].to_numpy(dtype=float)])
    return np.clip(x, -20, 20)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_or_load(directory, fit, validation, recipe, protocol):
    path = directory / "state.json"
    if path.exists():
        return read_json(path), 0
    directory.mkdir(parents=True, exist_ok=True)
    x = matrix(fit, recipe)
    state = {
        "recipe": recipe,
        "calibrated": recipe == "xgb_full",
        "fit_end": fit.target_time.max().isoformat(),
        "selection_end": validation.target_time.max().isoformat(),
    }
    if recipe == "xgb_full":
        model = XGBClassifier(
            n_estimators=protocol["xgb_estimators"],
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=50,
            reg_lambda=20,
            subsample=1,
            colsample_bytree=1,
            n_jobs=2,
            random_state=protocol["seed"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(x, fit.target)
        model.save_model(directory / "model.ubj")
        calibration = LogisticRegression(C=1, max_iter=500)
        calibration.fit(
            logit(model.predict_proba(matrix(validation, recipe))[:, 1]), validation.target
        )
        state.update(
            {
                "calibration_coef": float(calibration.coef_[0, 0]),
                "calibration_intercept": float(calibration.intercept_[0]),
            }
        )
        count = 2
    else:
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=protocol["logistic_c"], max_iter=1000, random_state=protocol["seed"]
        )
        model.fit(scaler.transform(x), fit.target)
        state.update(
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
            }
        )
        count = 1
    write_new_json(path, state)
    return state, count


def predict(directory, state, frame, *, raw=False):
    x = matrix(frame, state["recipe"])
    if state["recipe"] == "xgb_full":
        model = XGBClassifier()
        model.load_model(directory / "model.ubj")
        p = model.predict_proba(x)[:, 1]
        return (
            p
            if raw
            else sigmoid(
                logit(p).ravel() * state["calibration_coef"] + state["calibration_intercept"]
            )
        )
    return sigmoid(
        ((x - np.asarray(state["mean"])) / np.asarray(state["scale"])) @ np.asarray(state["coef"])
        + state["intercept"]
    )
