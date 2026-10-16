from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.dmn.data import FEATURES, Panel
from trading_ml.dmn.portfolio import metrics, simulate, target_weights
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)


class MomentumNetwork(nn.Module):
    """Unidirectional LSTM or small controls; no future timestep in the output."""

    def __init__(self, architecture: str, features: int, hidden: int, dropout: float):
        super().__init__()
        self.architecture = architecture
        self.input_dropout = nn.Dropout(dropout)
        if architecture == "lstm":
            self.encoder = nn.LSTM(features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        elif architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(features, hidden), nn.Tanh(), nn.Dropout(dropout)
            )
            self.head = nn.Linear(hidden, 1)
        elif architecture == "linear":
            self.encoder = nn.Identity()
            self.head = nn.Linear(features, 1)
        else:
            raise DataError(f"Arquitectura neuronal DMN inválida: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "lstm":
            encoded, _ = self.encoder(self.input_dropout(x))
            encoded = encoded[:, -1, :]
        else:
            encoded = self.encoder(self.input_dropout(x[:, -1, :]))
        return self.head(encoded).squeeze(-1)


def signal_from_output(output: torch.Tensor, objective: str, mode: str) -> torch.Tensor:
    if objective == "mse":
        signal = torch.sign(output)
        return signal.clamp(min=0) if mode == "long_only" else signal
    if objective == "binary":
        signal = torch.sign(output)  # logit threshold zero == probability 0.5
        return signal.clamp(min=0) if mode == "long_only" else signal
    signal = torch.tanh(output)
    # Affine mapping avoids the all-negative / zero-gradient dead zone of ReLU.
    return (signal + 1) / 2 if mode == "long_only" else signal


def differentiable_returns(
    signal: torch.Tensor,
    returns: torch.Tensor,
    volatility: torch.Tensor,
    risk: dict,
    costs: dict,
    *,
    net: bool,
    first_flat: bool,
) -> torch.Tensor:
    scale = risk["target_volatility"] / (
        volatility.clamp(min=risk["daily_volatility_floor"]) * np.sqrt(365)
    )
    # Use undivided per-asset weights. Sharpe is invariant to the fixed 1/N factor.
    weights = (signal * scale).clamp(-risk["maximum_gross"], risk["maximum_gross"])
    pnl = weights * returns
    if net:
        previous = torch.cat([torch.zeros_like(weights[:1]), weights[:-1]], dim=0)
        rate = sum(costs[k] for k in ["commission_bps", "slippage_bps", "half_spread_bps"]) / 10000
        pnl = pnl - rate * torch.abs(weights - previous)
        pnl = pnl - weights.clamp(max=0).abs() * costs["short_borrow_annual_bps"] / (10000 * 365)
    return pnl if first_flat else pnl[1:]


def sharpe_loss(returns: torch.Tensor, periods: int = 365, epsilon: float = 1e-8) -> torch.Tensor:
    """Equation 16: population moments pooled over asset/date; stabilized at zero."""
    mean = returns.mean()
    variance = ((returns - mean) ** 2).mean()
    return -mean * np.sqrt(periods) / torch.sqrt(variance + epsilon**2)


def fit_scaler(x: np.ndarray) -> dict:
    values = x.reshape(-1, x.shape[-1]).astype(np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        "clip": 5.0,
    }


def apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return np.ascontiguousarray(
        np.clip(
            (x - np.asarray(scaler["mean"])) / np.asarray(scaler["std"]),
            -scaler["clip"],
            scaler["clip"],
        ),
        dtype=np.float32,
    )


def predict_network(model: MomentumNetwork, x: np.ndarray, recipe: dict, risk: dict) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for first in range(0, len(x), 128):
            block = x[first : first + 128]
            output = model(torch.from_numpy(block.reshape(-1, *block.shape[-2:])))
            signal = signal_from_output(output, recipe["objective"], risk["position_mode"])
            predictions.append(signal.reshape(block.shape[:2]).numpy())
    return np.concatenate(predictions)


def fit_network(
    panel: Panel,
    train: np.ndarray,
    validation: np.ndarray,
    recipe: dict,
    protocol: dict,
    seed: int,
    directory: Path,
) -> tuple[MomentumNetwork, dict, dict]:
    t, risk, costs = (protocol[k] for k in ["training", "risk", "costs"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(t["threads"])
    torch.use_deterministic_algorithms(True)
    scaler = fit_scaler(panel.x[train])
    x = apply_scaler(panel.x[train], scaler)
    validation_x = apply_scaler(panel.x[validation], scaler)
    model = MomentumNetwork(recipe["architecture"], len(FEATURES), t["hidden_size"], t["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    y = torch.from_numpy(panel.returns[train].astype(np.float32))
    vol = torch.from_numpy(panel.volatility[train].astype(np.float32))
    tx = torch.from_numpy(x)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        starts = np.arange(0, len(train), t["batch_dates"])
        rng.shuffle(starts)  # Shuffle whole chronological blocks, never individual cost neighbors.
        losses = []
        for begin in starts:
            finish = min(begin + t["batch_dates"], len(train))
            lo = max(0, begin - 1)  # Include the true predecessor for turnover at a block edge.
            block = tx[lo:finish]
            output = model(block.reshape(-1, *block.shape[-2:])).reshape(block.shape[:2])
            objective = recipe["objective"]
            if objective == "mse":
                target = y[lo:finish] / vol[lo:finish]
                loss = nn.functional.mse_loss(output, target)
            elif objective == "binary":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output, (y[lo:finish] > 0).float()
                )
            else:
                signal = signal_from_output(output, objective, risk["position_mode"])
                pnl = differentiable_returns(
                    signal,
                    y[lo:finish],
                    vol[lo:finish],
                    risk,
                    costs,
                    net=recipe["net_training"],
                    first_flat=begin == 0,
                )
                loss = sharpe_loss(pnl) if objective == "sharpe" else -pnl.mean()
            if not torch.isfinite(loss):
                raise DataError("Pérdida DMN no finita; conservar corrida fallida")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), t["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            if not torch.isfinite(norm):
                raise DataError("Gradiente DMN no finito")
        signal = predict_network(model, validation_x, recipe, risk)
        w = target_weights(signal, panel.volatility[validation], risk)
        score = metrics(simulate(w, panel.returns[validation], panel.times[validation], costs))[
            "sharpe"
        ]
        score = score if score is not None else 0.0
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_net_portfolio_sharpe": score,
            }
        )
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = (
                copy.deepcopy(model.state_dict()),
                score,
                epoch,
                0,
            )
        else:
            stale += 1
        if stale >= t["patience"]:
            break
    if best_state is None:
        raise DataError("Entrenamiento sin checkpoint válido")
    model.load_state_dict(best_state)
    model.eval()
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, directory / "weights.pt")
    metadata = {
        "architecture": recipe["architecture"],
        "recipe": recipe,
        "seed": seed,
        "hidden_size": t["hidden_size"],
        "dropout": t["dropout"],
        "features": list(FEATURES),
        "scaler": scaler,
        "selected_epoch": best_epoch,
        "epochs_executed": len(history),
        "validation_selection": "net_portfolio_sharpe_after_costs",
        "validation_net_sharpe": best_score,
        "history": history,
        "train_first": panel.times[train[0]].isoformat(),
        "train_last_target": panel.target_times[train[-1]].isoformat(),
        "validation_last_target": panel.target_times[validation[-1]].isoformat(),
        "weights_sha256": sha256(directory / "weights.pt"),
        "dropout_note": "input/output dropout; no variational recurrent dropout",
    }
    write_new_json(directory / "metadata.json", metadata)
    LOG.info(
        "%s seed=%s epoch=%s validation Sharpe=%.3f", recipe["id"], seed, best_epoch, best_score
    )
    return model, scaler, metadata


def load_network(directory: Path) -> tuple[MomentumNetwork, dict]:
    meta = read_json(directory / "metadata.json")
    if sha256(directory / "weights.pt") != meta["weights_sha256"] or meta["features"] != list(
        FEATURES
    ):
        raise DataError("Checkpoint/contrato DMN alterado")
    model = MomentumNetwork(
        meta["architecture"], len(FEATURES), meta["hidden_size"], meta["dropout"]
    )
    model.load_state_dict(
        torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, meta


def fit_xgboost(
    panel: Panel,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    protocol: dict,
    seed: int,
    directory: Path,
) -> tuple[np.ndarray, dict]:
    from xgboost import XGBClassifier

    x = panel.x[:, :, -1, :]
    model = XGBClassifier(
        n_estimators=160,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=10,
        reg_lambda=10,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=15,
    )
    model.fit(
        x[train].reshape(-1, len(FEATURES)),
        (panel.returns[train] > 0).reshape(-1).astype(int),
        eval_set=[
            (
                x[validation].reshape(-1, len(FEATURES)),
                (panel.returns[validation] > 0).reshape(-1).astype(int),
            )
        ],
        verbose=False,
    )
    probability = model.predict_proba(x[test].reshape(-1, len(FEATURES)))[:, 1].reshape(
        len(test), -1
    )
    signal = np.sign(probability - 0.5)
    if protocol["risk"]["position_mode"] == "long_only":
        signal = signal.clip(0, 1)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_model(directory / "model.ubj")
    meta = {
        "seed": seed,
        "architecture": "xgboost",
        "features": list(FEATURES),
        "selected_trees": int(model.best_iteration + 1),
        "validation_selection": "logloss; supervised control differs from Sharpe checkpoint selection",
        "train_last_target": panel.target_times[train[-1]].isoformat(),
        "validation_last_target": panel.target_times[validation[-1]].isoformat(),
        "weights_sha256": sha256(directory / "model.ubj"),
    }
    write_new_json(directory / "metadata.json", meta)
    return signal, meta
