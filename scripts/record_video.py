import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO, A2C

from env.flappy_bird_simple import FlappyBirdSimpleEnv


def record_video(model_path: str, output_path: str = "videos/agent.mp4", max_frames: int = 50000):
    # Detect algorithm type from filename
    if "ppo" in model_path.lower():
        ModelClass = PPO
    elif "a2c" in model_path.lower():
        ModelClass = A2C
    else:
        raise ValueError("Cannot determine algorithm type from model path.")

    print(f"Loading model from: {model_path}")
    model = ModelClass.load(model_path)

    # Create environment with rgb_array render mode
    env = FlappyBirdSimpleEnv(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []

    # Collect frames
    for _ in range(max_frames):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Render as rgb array and save frame
        frame = env._render_rgb_array()
        frames.append(frame)

        if terminated:
            break

    env.close()

    print(f"Saving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=30)
    print("Video saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a video of a trained PPO/A2C agent.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g., saved_models/flappy_ppo_final.zip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videos/agent.mp4",
        help="Output video filepath.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Maximum number of frames to record.",
    )

    args = parser.parse_args()

    record_video(args.model, args.output, args.max_frames)

# final model
# python -m scripts.record_video --model saved_models/flappy_ppo_final.zip

# best model
# python -m scripts.record_video --model saved_models/flappy_ppo_best/best_model.zip

#
