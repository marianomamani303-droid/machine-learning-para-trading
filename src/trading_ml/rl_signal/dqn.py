from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from trading_ml.common import write_new_json
from trading_ml.rl_signal.environment import TradingEnvironment, rollout


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray
    age_scale: int

    @classmethod
    def fit(cls, train: pd.DataFrame, config: dict):
        p = config["portfolio"]
        cost = (p["commission_bps"] + p["half_spread_bps"] + p["slippage_bps"]) / 10000
        x = np.column_stack([train.probability, train.sigma, np.full(len(train), cost)])
        scaler = StandardScaler().fit(x)
        return cls(scaler.mean_, scaler.scale_, p["age_scale_days"])

    def transform(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).copy()
        result[..., :3] = (result[..., :3] - self.mean) / self.scale
        result[..., 3] /= 0.5
        result[..., 4] /= self.age_scale
        return np.ascontiguousarray(result, dtype=np.float32)

    def as_dict(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "age_scale": self.age_scale,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(np.array(value["mean"]), np.array(value["scale"]), value["age_scale"])


def network(config: dict):
    h1, h2 = config["training"]["hidden"]
    return nn.Sequential(
        nn.Linear(5, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 3)
    )


def fit_dqn(train, validation, config: dict, seed: int, output: Path):
    """Only the train environment can add transitions or call optimizer.step()."""
    output.mkdir(parents=True, exist_ok=False)
    t = config["training"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online, target = network(config), network(config)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=t["learning_rate"])
    norm = Normalizer.fit(train, config)
    write_new_json(output / "normalizer.json", norm.as_dict())
    env = TradingEnvironment(train, config)
    state = norm.transform(env.reset())
    capacity = t["replay_capacity"]
    states = np.zeros((capacity, 5), np.float32)
    next_states = np.zeros_like(states)
    actions = np.zeros(capacity, np.int64)
    rewards = np.zeros(capacity, np.float32)
    dones = np.zeros(capacity, np.float32)
    masks = np.zeros((capacity, 3), bool)
    dates = np.zeros(capacity, np.int64)
    count, episodes = 0, 0
    best_score, best_step, best_weights = -np.inf, None, None
    records = []
    losses = []
    for step in range(1, t["maximum_steps"] + 1):
        epsilon = t["epsilon_start"] + (t["epsilon_end"] - t["epsilon_start"]) * min(
            1.0, (step - 1) / t["epsilon_decay_steps"]
        )
        if rng.random() < epsilon:
            action = int(rng.choice(np.flatnonzero(env.mask())))
        else:
            with torch.no_grad():
                values = online(torch.from_numpy(state[None])).numpy()[0]
            action = int(np.argmax(np.where(env.mask(), values, -np.inf)))
        training_row = env.i
        observation, reward, done = env.step(action)
        next_state = norm.transform(observation)
        slot = count % capacity
        states[slot], next_states[slot] = state, next_state
        actions[slot], rewards[slot], dones[slot] = action, reward * t["reward_scale"], done
        masks[slot] = env.mask()
        dates[slot] = training_row
        count += 1
        if step >= t["warmup_steps"] and step % t["update_every"] == 0:
            batch = rng.choice(min(count, capacity), size=t["batch_size"], replace=False)
            with torch.no_grad():
                q_next = target(torch.from_numpy(next_states[batch]))
                q_next[~torch.from_numpy(masks[batch])] = -torch.inf
                expected = (
                    torch.from_numpy(rewards[batch])
                    + t["gamma"] * (1 - torch.from_numpy(dones[batch])) * q_next.max(dim=1).values
                )
            actual = (
                online(torch.from_numpy(states[batch]))
                .gather(1, torch.from_numpy(actions[batch, None]))
                .squeeze(1)
            )
            loss = nn.functional.smooth_l1_loss(actual, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), t["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        if step % t["target_every"] == 0:
            target.load_state_dict(online.state_dict())
        if done:
            episodes += 1
            state = norm.transform(env.reset())
        else:
            state = next_state
        if step % t["checkpoint_every"] == 0:
            online.eval()
            curve, orders = rollout(validation, config, "dqn", normalizer=norm, network=online)
            score = float(curve.reward.sum())
            record = {
                "step": step,
                "episodes_completed": episodes,
                "epsilon": epsilon,
                "mean_training_huber": float(np.mean(losses)) if losses else None,
                "validation_reward": score,
                "validation_log_return": float(curve.log_return.sum()),
                "validation_orders": int(curve.filled_orders.sum()),
            }
            records.append(record)
            # All six checkpoints retained, including RNG/optimizer for forensic replay.
            torch.save(
                {
                    "online": online.state_dict(),
                    "target": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "seed": seed,
                    "rng_state": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                },
                output / f"step_{step:05d}.pt",
            )
            curve.to_csv(output / f"validation_{step:05d}.csv", index=False)
            orders.to_csv(output / f"validation_orders_{step:05d}.csv", index=False)
            if score > best_score:
                best_score, best_step = score, step
                best_weights = {k: v.detach().clone() for k, v in online.state_dict().items()}
            online.train()
    online.load_state_dict(best_weights)
    online.eval()
    torch.save(online.state_dict(), output / "selected.pt")
    np.savez_compressed(
        output / "training_replay.npz",
        states=states[: min(count, capacity)],
        next_states=next_states[: min(count, capacity)],
        actions=actions[: min(count, capacity)],
        rewards=rewards[: min(count, capacity)],
        dones=dones[: min(count, capacity)],
        masks=masks[: min(count, capacity)],
        row_indices=dates[: min(count, capacity)],
    )
    summary = {
        "seed": seed,
        "selected_step": best_step,
        "selected_validation_reward": best_score,
        "steps": t["maximum_steps"],
        "unique_training_dates": len(train),
        "completed_episodes": episodes,
        "replay_transitions": min(count, capacity),
        "replay_training_only": True,
        "checkpoints": records,
    }
    write_new_json(output / "training.json", summary)
    return online, norm, summary
