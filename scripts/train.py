import os
import argparse
from typing import Any, Dict
import time

import gymnasium as gym  # for typing only
import yaml
import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from env.flappy_bird_simple import FlappyBirdSimpleEnv
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------
# LOG HYPERPARAMETERS INTO TENSORBOARD (TABLE + SCALARS)
# ---------------------------------------------------------------------
def log_hyperparams(writer: SummaryWriter, params: Dict[str, Any]):
    """Logs hyperparameters cleanly into TensorBoard."""
    # Table-style log
    table = "Hyperparameters Used:\n-------------------\n"
    for k, v in params.items():
        table += f"{k}: {v}\n"

    writer.add_text("hyperparams_table", f"```\n{table}\n```", 0)

    # Also log individual scalars
    for key, value in params.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"hyperparams/{key}", value, 0)
        else:
            writer.add_text(f"hyperparams/{key}", str(value), 0)


# ---------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# ENV FACTORY (MAC-SAFE)
# ---------------------------------------------------------------------
def make_env(seed: int | None = None, render_mode: str | None = None):
    """Creates env wrapped with Monitor, safe for DummyVecEnv."""

    def _init():
        env = FlappyBirdSimpleEnv(render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init


# ---------------------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------------------
def make_model(
    algorithm: str,
    policy: str,
    env: DummyVecEnv,
    hyperparams: Dict[str, Any],
    tensorboard_log: str,
):
    """Creates PPO or A2C from config."""
    algo = algorithm.upper()

    kwargs = dict(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device="cpu",  # Mac-safe
    )

    # Clean hyperparameters (remove non-SB3 keys)
    h = hyperparams.copy()
    h.pop("algorithm", None)
    h.pop("policy", None)

    kwargs.update(h)

    if algo == "PPO":
        return PPO(**kwargs)
    elif algo == "A2C":
        return A2C(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


# ---------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------
def train(config_path: str, seed: int | None = 42, total_timesteps_override: int | None = None):
    # Load config
    config = load_config(config_path)

    # Extract hyperparams
    h = config.get("hyperparameters", config.get("hyperparameter", {}))
    algorithm = h.get("algorithm", "PPO")
    policy = h.get("policy", "MlpPolicy")

    total_timesteps = total_timesteps_override or config.get(
        "total_timesteps", 500_000)
    eval_freq = config.get("eval_freq", 10_000)

    ckpt_cfg = config.get("checkpoints", {})
    ckpt_prefix = ckpt_cfg.get("prefix", f"flappy_{algorithm.lower()}")

    # Create folders
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # UNIQUE RUN NAME FOR TENSORBOARD
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{ckpt_prefix}_run_{timestamp}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer (for hyperparams)
    writer = SummaryWriter(log_dir)
    log_hyperparams(writer, h)

    # ------------------------------------------------------------------
    # ENVIRONMENTS
    # ------------------------------------------------------------------
    train_env = DummyVecEnv([make_env(seed=seed, render_mode=None)])
    eval_env = DummyVecEnv(
        [make_env(seed=(seed + 1) if seed else None, render_mode=None)])

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    model = make_model(
        algorithm=algorithm,
        policy=policy,
        env=train_env,
        hyperparams=h,
        tensorboard_log=log_dir,
    )

    # ------------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------------
    callbacks = []

    # Eval callback
    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(
                "saved_models", f"{ckpt_prefix}_best"),
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
    )

    # Checkpoint callback
    callbacks.append(
        CheckpointCallback(
            save_freq=eval_freq,
            save_path="saved_models",
            name_prefix=ckpt_prefix,
        )
    )

    callback_list = CallbackList(callbacks)

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    print(
        f"🚀 Starting training ({algorithm}) for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    writer.close()

    # ------------------------------------------------------------------
    # SAVE FINAL MODEL
    # ------------------------------------------------------------------
    final_path = os.path.join("saved_models", f"{ckpt_prefix}_final")
    model.save(final_path)

    print(f"🎉 Training finished! Final model saved to: {final_path}")
    print(f"📊 TensorBoard logs saved to: {log_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO/A2C agent on FlappyBirdSimpleEnv.")
    parser.add_argument("--config", type=str, default="config/simple_ppo.yml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=None)

    args = parser.parse_args()
    seed_arg = None if args.seed == -1 else args.seed

    train(args.config, seed_arg, args.total_timesteps)


# running the script
# python scripts/train.py --config config/simple_ppo.yml

# tesnor board initialization
# tensorboard --logdir logs
