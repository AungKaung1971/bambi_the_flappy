import os
import argparse
from typing import Any, Dict

import gymnasium as gym  # for typing only, env is custom
import yaml
import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from env.flappy_bird_simple import FlappyBirdSimpleEnv


# ------------------------------------------------------------
# CONFIG LOADING
# ------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ------------------------------------------------------------
# ENV FACTORIES (NO MULTIPROCESSING, MAC-SAFE)
# ------------------------------------------------------------
def make_env(seed: int | None = None, render_mode: str | None = None):
    """
    Returns a function that creates a single FlappyBirdSimpleEnv wrapped with Monitor,
    suitable for DummyVecEnv.
    """

    def _init():
        env = FlappyBirdSimpleEnv(render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
        env = Monitor(env)
        return env

    return _init


# ------------------------------------------------------------
# MODEL FACTORY
# ------------------------------------------------------------
def make_model(
    algorithm: str,
    policy: str,
    env: DummyVecEnv,
    hyperparams: Dict[str, Any],
    tensorboard_log: str | None = None,
):
    algo = algorithm.upper()
    algo_kwargs = dict(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device="cpu",  # enforce CPU for Mac safety
    )

    # Remove keys that are not SB3 kwargs
    hyperparams = hyperparams.copy()
    hyperparams.pop("algorithm", None)
    hyperparams.pop("policy", None)

    algo_kwargs.update(hyperparams)

    if algo == "PPO":
        model_cls = PPO
    elif algo == "A2C":
        model_cls = A2C
    else:
        raise ValueError(f"Unsupported algorithm in config: {algorithm}")

    model = model_cls(**algo_kwargs)
    return model


# ------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------------------------------------
def train(config_path: str, seed: int | None = 42, total_timesteps_override: int | None = None):
    # Load config
    config = load_config(config_path)

    # Read top-level config fields
    hyper = config.get("hyperparameters", config.get("hyperparameter", {}))
    algorithm = hyper.get("algorithm", "PPO")
    policy = hyper.get("policy", "MlpPolicy")

    total_timesteps = total_timesteps_override or config.get(
        "total_timesteps", 500_000)
    eval_freq = config.get("eval_freq", 10_000)

    checkpoints_cfg = config.get("checkpoints", {})
    ckpt_prefix = checkpoints_cfg.get("prefix", f"flappy_{algorithm.lower()}")

    # Make sure directories exist
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Seed
    if seed is not None:
        np.random.seed(seed)

    # --------------------------------------------------------
    # CREATE TRAINING ENV (DummyVecEnv, NO SubprocVecEnv)
    # --------------------------------------------------------
    train_env = DummyVecEnv([make_env(seed=seed, render_mode=None)])

    # --------------------------------------------------------
    # CREATE EVAL ENV (separate, no rendering)
    # --------------------------------------------------------
    eval_env = DummyVecEnv(
        [make_env(seed=seed + 1 if seed is not None else None, render_mode=None)])

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    model = make_model(
        algorithm=algorithm,
        policy=policy,
        env=train_env,
        hyperparams=hyper,
        tensorboard_log="logs",
    )

    # --------------------------------------------------------
    # CALLBACKS: EVAL + CHECKPOINTS
    # --------------------------------------------------------
    callbacks = []

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(
            "saved_models", f"{ckpt_prefix}_best"),
        log_path="logs",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Optional checkpoint callback (saves periodic snapshots)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="saved_models",
        name_prefix=ckpt_prefix,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    print(
        f"Starting training with {algorithm} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback_list)

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------
    final_model_path = os.path.join("saved_models", f"{ckpt_prefix}_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to: {final_model_path}")


# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO/A2C agent on FlappyBirdSimpleEnv.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/simple_ppo.yml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (set -1 for no fixed seed).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config if provided.",
    )

    args = parser.parse_args()

    seed_arg = None if args.seed == -1 else args.seed

    train(
        config_path=args.config,
        seed=seed_arg,
        total_timesteps_override=args.total_timesteps,
    )


# running the script
# python scripts/train.py --config config/simple_ppo.yml
