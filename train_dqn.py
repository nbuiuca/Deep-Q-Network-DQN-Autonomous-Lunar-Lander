import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# settings
ENV = "LunarLander-v3"
EPISODES = 600
BATCH = 64
MEMORY_SIZE = 25000
GAMMA = 0.99
LR = 0.001
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 400
TARGET_UPDATE = 1000

# "human"  live Lunar Lander window while training; None = no window (faster)
RENDER_MODE = "human"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, a)
        )

    def forward(self, x):
        return self.net(x)

# Epsilon function
def get_epsilon(ep):
    if ep >= EPS_DECAY:
        return EPS_END
    return EPS_START - (ep / EPS_DECAY) * (EPS_START - EPS_END)


def train():
    env = gym.make(ENV, render_mode=RENDER_MODE) if RENDER_MODE else gym.make(ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = DQN(state_size, action_size).to(device)
    target = DQN(state_size, action_size).to(device)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)

    steps = 0

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        epsilon = get_epsilon(ep)

        while True:
            # choose action based on policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).to(device)
                    action = policy(s).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            # learn
            if len(memory) >= BATCH:
                batch = random.sample(memory, BATCH)

                states = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(device)
                actions = torch.tensor([b[1] for b in batch]).to(device)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
                next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

                q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target(next_states).max(1)[0]
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if steps % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            if done or truncated:
                break

        print(f"Episode {ep+1}: Reward = {total_reward:.1f}, Epsilon = {epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    train()