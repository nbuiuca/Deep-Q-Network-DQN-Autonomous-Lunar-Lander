# Deep Q-Network (DQN) Autonomous Lunar Lander

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A7?style=flat-square)
![Reinforcement Learning](https://img.shields.io/badge/RL-Deep_Q--Learning-10B981?style=flat-square)

An end-to-end **Deep Reinforcement Learning** agent built from scratch in **PyTorch** to solve the continuous-state spacecraft trajectory control problem in Gymnasium's `LunarLander-v3` physics environment. Without prior domain rules, the agent learns optimal thruster firing policies to land safely between the target flags through trial-and-error interaction.

---

## Key Technical Features

* **3-Layer MLP Q-Network Architecture:** Approximates continuous-state action-value functions `Q(s, a)` using a feedforward neural network (`Input(8) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Output(4)`).
* **Experience Replay Buffer:** Stores up to **25,000** state transitions `(state, action, reward, next_state, done)` in a circular `deque` and samples random mini-batches of **64** transitions per step, breaking temporal correlations between consecutive frames.
* **Decoupled Target Network:** Maintains a separate `target` network whose weights are synchronized with the active `policy` network every **1,000 steps**, preventing moving-target oscillations during Bellman updates.
* **Linear Epsilon-Greedy Exploration:** Balances exploration and exploitation by decaying `epsilon` linearly from **1.00** (pure random exploration) down to **0.05** (95% optimal policy exploitation) over the first **400 episodes**.
* **Hardware Acceleration & Live Rendering:** Automatically detects and utilizes **NVIDIA CUDA** GPUs when available, with a toggleable `RENDER_MODE` for real-time 2D physics visualization (`"human"`) or high-speed headless training (`None`).

---

## Environment Specifications (`LunarLander-v3`)

### 1. Observation Space (`8-Dimensional Continuous Vector`)
At each timestep, the environment provides an 8-element state vector `s`:

| Index | Observation | Description |
| :--- | :--- | :--- |
| `0` | `x` | Horizontal coordinate relative to the landing pad center |
| `1` | `y` | Vertical coordinate relative to the landing pad |
| `2` | `v_x` | Horizontal linear velocity |
| `3` | `v_y` | Vertical linear velocity |
| `4` | `theta` | Spacecraft orientation angle |
| `5` | `omega` | Angular velocity |
| `6` | `left_leg` | Boolean (`1.0` or `0.0`) indicating left leg ground contact |
| `7` | `right_leg` | Boolean (`1.0` or `0.0`) indicating right leg ground contact |

### 2. Action Space (`4 Discrete Actions`)
| Action ID | Thruster Command |
| :--- | :--- |
| `0` | Do nothing (coast) |
| `1` | Fire left orientation engine |
| `2` | Fire main bottom engine |
| `3` | Fire right orientation engine |

### 3. Reward Dynamics
* Moving from the top of the screen to the landing pad at zero speed yields **+100 to +140 points**.
* Coming to rest safely on the pad grants an additional **+100 points**; crashing results in **-100 points**.
* Each leg making ground contact grants **+10 points**.
* Fuel penalty: firing the main engine costs **-0.3 points/frame**; side engines cost **-0.03 points/frame**.
* An episode is considered **solved** upon achieving a total score of **200+ points**.

---

## Training Algorithm & Hyperparameters

The policy network minimizes the Mean Squared Error (MSE) loss between predicted Q-values and the target Bellman equation:

```text
Q_target = r + gamma * max_a' Q_target_net(s', a') * (1 - done)