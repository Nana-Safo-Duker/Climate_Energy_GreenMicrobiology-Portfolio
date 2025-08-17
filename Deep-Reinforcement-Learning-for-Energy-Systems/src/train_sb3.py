from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

try:
    from drl_energy_env import MultiEnergyToyEnv
except ImportError:  # python -m src.train_sb3 from project root
    from src.drl_energy_env import MultiEnergyToyEnv


def evaluate(model: PPO, episodes: int = 5, seed: int = 0) -> dict[str, float]:
    returns = []
    costs = []
    comfort = []
    for ep in range(episodes):
        env = Monitor(MultiEnergyToyEnv(seed=seed + ep))
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_cost = 0.0
        ep_comfort = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_cost += float(info.get("step_cost", 0.0))
            ep_comfort += float(info.get("comfort_violation", 0.0))
        returns.append(ep_return)
        costs.append(ep_cost)
        comfort.append(ep_comfort)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_cost": float(np.mean(costs)),
        "mean_comfort_violation": float(np.mean(comfort)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=505)
    p.add_argument("--outdir", type=str, default=str(Path("outputs") / "models"))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = Monitor(MultiEnergyToyEnv(seed=args.seed))

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model_path = outdir / "ppo_multi_energy_toy.zip"
    model.save(model_path)

    metrics = evaluate(model, episodes=5, seed=args.seed + 100)
    metrics_path = outdir / "evaluation_metrics.txt"
    metrics_path.write_text(
        "\n".join([f"{k}={v}" for k, v in metrics.items()]) + "\n", encoding="utf-8"
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()

