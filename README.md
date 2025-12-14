🐦 Bambi the Flappy — Reinforcement Learning Agent
Overview

Bambi the Flappy is a custom-built reinforcement learning project in which a PPO (Proximal Policy Optimization) agent learns to play a Flappy Bird–style game from scratch.

The project combines:

A Gymnasium-compatible environment built in Pygame

Policy-gradient reinforcement learning using Stable-Baselines3

A config-driven training pipeline with reproducible experiments

TensorBoard-based logging for performance, rewards, and hyperparameters

A visualized inference mode with raycasts and HUD overlays

The goal is not just to recreate Flappy Bird, but to study how reward shaping, environment design, and PPO hyperparameters influence agent behaviour.

Key Features

✅ Custom Flappy Bird environment (Gymnasium + Pygame)

✅ PPO agent trained with Stable-Baselines3

✅ Configurable reward structure via YAML

✅ Deterministic evaluation & best-model checkpointing

✅ TensorBoard logging (metrics + hyperparameters)

✅ Gameplay recording to MP4

✅ Engineering-style HUD with raycasts and state visualization

Environment Design
Observation Space

The agent observes a low-dimensional state vector:

[ bird_y,
  bird_velocity,
  distance_to_next_pipe,
  gap_top_y,
  gap_bottom_y ]


This design forces the agent to learn control and timing, rather than memorizing pixels.

Action Space

Discrete:

0 — do nothing

1 — flap

Episode Termination

An episode ends when:

The bird collides with a pipe

The bird exits the vertical bounds of the screen

Reward Function

The reward function is explicitly configurable via YAML and was iteratively tuned during development.

Typical components include:

Survival reward (per timestep)

Pipe-passing bonus

Terminal penalty on death

Example (configurable):

reward:
  survival: 1.0
  pipe: 10.0
  death: -100.0


This setup encourages:

Long-term survival

Clean pipe traversal

Avoidance of reckless behaviour

Training Setup

Algorithm: PPO (Proximal Policy Optimization)

Policy: MLP (fully connected)

Framework: Stable-Baselines3

Training Duration: up to 500,000 timesteps per run

Evaluation: Periodic deterministic rollouts

Hardware: CPU-only (Mac-safe)

All hyperparameters and rewards are logged per run.

Results & Metrics

📊 Metrics below are representative — exact values depend on configuration.

Tracked during training:

Mean episode length

Mean episode reward

Pipes passed per episode

Best model checkpoint performance

The trained agent is capable of surviving hundreds of pipes in a fixed-difficulty environment.

(TensorBoard plots and gameplay videos should be linked here)

Visualization & Debugging

During inference and recording, the environment renders:

Raycasts from bird → pipe gap (top / center / bottom)

Distance labels

Velocity and alignment HUD

Real-time score tracking

This makes the agent’s decision-making observable and explainable, rather than a black box.

How to Run
Train the Agent
python scripts/train.py --config config/simple_ppo.yml

View Training Metrics
tensorboard --logdir logs

Record Gameplay
python scripts/record_video.py \
  --model saved_models/flappy_ppo_best/best_model.zip

Project Structure
bambi_the_flappy/
├── env/                # Custom Gymnasium environment
├── scripts/            # Training, evaluation, recording
├── config/             # YAML configs (hyperparams & rewards)
├── logs/               # TensorBoard runs
├── saved_models/       # Checkpoints & best models
└── videos/             # Recorded gameplay

Design Decisions

Why PPO?
Stable, on-policy algorithm well-suited for continuous control and sparse rewards.

Why not CNNs?
Low-dimensional state representation allows faster learning and interpretability.

Why custom reward shaping?
Raw survival alone leads to unstable policies; shaping accelerates convergence.

Limitations

No difficulty scaling (fixed pipe speed & gap)

Deterministic physics (no domain randomization)

MLP-only policy (no vision-based agent)

These are deliberate to isolate learning dynamics.

Future Work

Dynamic difficulty scaling

Gap-alignment reward shaping

Curriculum learning

CNN-based visual policy

Domain randomization for robustness

Why This Project Matters

This project demonstrates:

End-to-end RL system design

Experimental rigor (configs, metrics, reproducibility)

Practical debugging and visualization

Clear separation between environment, agent, and training logic

It is not a tutorial clone — it is an engineering experiment.