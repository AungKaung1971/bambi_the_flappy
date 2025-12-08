import argparse
import time

from stable_baselines3 import PPO, A2C
from env.flappy_bird_simple import FlappyBirdSimpleEnv


def evaluate(model_path: str, episodes: int = 5):
    # Detect algorithm type from filename
    if "ppo" in model_path.lower():
        ModelClass = PPO
    elif "a2c" in model_path.lower():
        ModelClass = A2C
    else:
        raise ValueError("Cannot determine algorithm type from model path.")

    print(f"Loading model from: {model_path}")
    model = ModelClass.load(model_path)

    env = FlappyBirdSimpleEnv(render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        ep_reward = 0

        while not terminated:
            # Model predicts action
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            time.sleep(1 / 30)  # match render FPS (smooth playback)

        print(f"Episode {ep+1} reward: {ep_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO/A2C agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g., saved_models/flappy_ppo_final.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many episodes to run.",
    )

    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes)
