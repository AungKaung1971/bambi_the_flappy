🐦 Bambi the Flappy — Reinforcement Learning Agent
==================================================

Overview
--------

**Bambi the Flappy** is a reinforcement learning project where a custom-built PPO (Proximal Policy Optimization) agent learns to master a Flappy Bird–style game — from scratch.

This project integrates:

*   A **Gymnasium-compatible environment** built in **Pygame**
    
*   **Policy-gradient RL** via **Stable-Baselines3**
    
*   **Config-driven training pipelines** for reproducible experimentation
    
*   **TensorBoard** logging for metrics and hyperparameters
    
*   A **visualized inference mode** with raycasts and HUD overlays
    

The goal is not to merely clone Flappy Bird, but to investigate how **reward shaping**, **environment design**, and **PPO hyperparameters** impact agent behavior.

🧠 Key Features
---------------

*   ✅ Custom Flappy Bird environment (Gymnasium + Pygame)
    
*   ✅ PPO agent via Stable-Baselines3
    
*   ✅ YAML-based reward configuration
    
*   ✅ Deterministic evaluation & checkpointing
    
*   ✅ Training visualization via TensorBoard
    
*   ✅ Gameplay recording to MP4
    
*   ✅ In-game HUD with real-time raycasts and debug info
    

📁 Table of Contents
--------------------

*   Installation
    
*   Environment Design
    
*   Reward Function
    
*   Training Setup
    
*   Results & Metrics
    
*   Visualization & Debugging
    
*   How to Run
    
*   Project Structure
    
*   Design Decisions
    
*   Limitations
    
*   Future Work
    
*   Why This Project Matters
    
*   License
    

🔧 Installation
---------------

**Requirements:**

*   Python ≥ 3.8
    
*   pygame
    
*   gymnasium
    
*   stable-baselines3
    
*   tensorboard
    
*   opencv-python
    
*   PyYAML
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

> ℹ️ Mac users: The entire pipeline is CPU-safe and does **not** require a GPU.

🎮 Environment Design
---------------------

### **Observation Space**

A 5-dimensional state vector:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [    bird_y,    bird_velocity,    distance_to_next_pipe,    gap_top_y,    gap_bottom_y  ]   `

### **Action Space**

*   0: Do nothing
    
*   1: Flap
    

### **Episode Termination**

An episode ends when:

*   The bird hits a pipe
    
*   The bird flies off-screen vertically
    

🎯 Reward Function
------------------

The reward structure is **YAML-configurable** and promotes:

*   Long-term survival
    
*   Clean pipe traversal
    
*   Risk-averse decision-making
    

### **Sample Reward Config:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   reward:    survival: 1.0    pipe: 10.0    death: -100.0   `

This balance was tuned iteratively to stabilize learning.

🏋️ Training Setup
------------------

*   **Algorithm:** PPO (Proximal Policy Optimization)
    
*   **Policy:** MLP (Fully Connected)
    
*   **Library:** Stable-Baselines3
    
*   **Timesteps:** Up to 500,000 per run
    
*   **Hardware:** CPU-only
    
*   **Evaluation:** Deterministic rollouts with checkpointing
    
*   **Logging:** TensorBoard (metrics + hyperparameters)
    

📊 Results & Metrics
--------------------

Tracked during training:

*   Mean episode reward
    
*   Mean episode length
    
*   Pipes passed per episode
    
*   Best checkpoint performance
    

> The trained agent can survive hundreds of pipes in a stable, fixed-difficulty setup.

📈 _(Include TensorBoard screenshots or links here)_🎥 _(Link to gameplay recordings in /videos)_

🧪 Visualization & Debugging
----------------------------

During inference, the game renders:

*   Raycasts from the bird to the pipe gap (top / center / bottom)
    
*   Velocity, alignment, and distance overlays
    
*   Real-time score HUD
    

This transforms the agent from a black box into an **explainable** decision system.

▶️ How to Run
-------------

### **Train the Agent**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/train.py --config config/simple_ppo.yml   `

### **View Training Metrics**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   tensorboard --logdir logs   `

### **Record Gameplay**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/record_video.py \    --model saved_models/flappy_ppo_best/best_model.zip   `

🗂️ Project Structure
---------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bambi_the_flappy/  ├── env/                # Custom Gymnasium environment  ├── scripts/            # Training, evaluation, recording  ├── config/             # YAML configs (hyperparams & rewards)  ├── logs/               # TensorBoard logs  ├── saved_models/       # Trained checkpoints  └── videos/             # Recorded gameplay videos   `

🧠 Design Decisions
-------------------

### Why PPO?

Stable on-policy algorithm, ideal for sparse rewards and continuous control.

### Why MLP (not CNN)?

The low-dimensional input is structured; no need for image-based learning.

### Why Custom Rewards?

Pure survival leads to erratic behavior. Shaped rewards accelerate convergence and stabilize training.

⚠️ Limitations
--------------

*   No dynamic difficulty (pipes & gaps are fixed)
    
*   No domain randomization (deterministic physics)
    
*   No vision-based agent (MLP only, not CNN)
    

> These are **intentional** to focus purely on learning dynamics and PPO behavior.

🚀 Future Work
--------------

*   Dynamic difficulty adjustment
    
*   Gap-alignment reward shaping
    
*   Curriculum learning
    
*   CNN-based visual policy
    
*   Domain randomization for robustness
    

💡 Why This Project Matters
---------------------------

This is not a tutorial clone — it’s an **engineering experiment** in end-to-end reinforcement learning. It demonstrates:

*   Modular RL system design
    
*   Experimental reproducibility with config-driven runs
    
*   Real-time explainability and agent visualization
    
*   Clean separation of agent, environment, and training logic
    

📜 License
----------

MIT License. See LICENSE file for details.
