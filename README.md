# AI Learned Flappy Bird (DQN)

A Deep Q-Network (DQN) reinforcement learning agent trained to play Flappy Bird using PyTorch and [`flappy-bird-gymnasium`](https://github.com/Talendar/flappy-bird-gymnasium).

## Features

- **Deep Q-Network (DQN)**: A neural network that estimates the expected return, or Q-value, for each possible action in a given game state. In Flappy Bird, the network learns to predict whether flapping or doing nothing is more valuable for future success.
- **How DQNs Work**: The agent observes the current state of the environment, feeds it into the network, and picks the action with the highest estimated Q-value. It then receives a reward, stores the transition, and updates the network using the difference between the predicted value and the target value from future rewards.
- **Experience Replay**: The agent stores past transitions such as state, action, reward, and next state in a replay buffer. During training, random samples are drawn from this buffer so the network learns from a more diverse set of experiences instead of only the most recent, highly correlated steps.
- **Epsilon-Greedy Strategy**: At the start of training, the agent explores by taking random actions with a high probability. As training progresses, the epsilon value decays, so the agent increasingly exploits its learned policy and chooses the best action according to the network.
- **Target Network**: A separate network is periodically synced with the main network to provide more stable target Q-values during training, reducing oscillations and helping the agent learn reliably.
- **Model Checkpointing**: Automatic best-model saving and resuming from saved checkpoints.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Madhura-Kedar/Ai-learned-Flappy-Bird.git
cd Ai-learned-Flappy-Bird
```

2. Install dependencies:
```bash
pip install gymnasium flappy-bird-gymnasium pygame torch pyyaml
```

## Usage

### 1. Train the Agent
To train the DQN agent using configuration parameters defined in `parameters.yaml`:
```bash
python agent.py flappybirdv0 --train
```

### 2. Test / Evaluate the Agent
To watch the trained agent play with visual rendering:
```bash
python agent.py flappybirdv0
```

### 3. Play Manually
To play the game manually using the `Space` key:
```bash
python game_flappy_bird.py
```

## Project Structure

```
├── agent.py               # Main agent training and evaluation logic
├── dqn.py                 # PyTorch DQN neural network architecture
├── experience_replay.py   # Replay memory buffer implementation
├── game_flappy_bird.py    # Manual play script
├── parameters.yaml        # Hyperparameters (learning rate, discount, epsilon, etc.)
└── runs/                  # Saved models and training logs (gitignored)
```
