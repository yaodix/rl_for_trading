gymnasium = 1.2.3


#### ubuntu查看ROM的路径

/home/yao/anaconda3/envs/py310/lib/python3.10/site-packages/AutoROM/roms

### 测试失败

ch06_atari_env_test.py 测试失败,报错gymnasium.error.NameNotFound: Environment `pong` doesn't exist

    所有环境安装完备，问题可能是gym没有识别到，解决方法是import ale_py


## Gymnasium 教程以及安装、常见报错解决\_gymnasium安装-CSDN博客

本教程详细指导你在 Windows 的 Conda 环境中配置 `gymnasium==1.1.1`，使用 Atari 环境（`PongNoFrameskip-v4`）进行强化学习实验，包含深度 Q 网络（DQN）和近端策略优化（PPO）两种算法。本教程针对初学者，解决常见问题（如 ROM 配置、`box2d-py` 编译、环境 ID 错误），并提供帧堆叠、图像预处理和训练视频保存等实用功能。

### 前提条件

* **操作系统**：Windows 10/11（64 位）
* **Conda**：已安装 Anaconda 或 Miniconda（推荐 Anaconda 2023.09 或更新版本）
* **Python 基础**：熟悉 Python 编程
* **强化学习知识**：了解 Q 学习、策略梯度等基本概念（可选）
* **硬件**：建议至少 8GB 内存，GPU 可加速训练（本教程以 CPU 为例）

### 第一步：创建并激活 [Conda 环境](https://so.csdn.net/so/search?q=Conda%20%E7%8E%AF%E5%A2%83&spm=1001.2101.3001.7020)

为避免依赖冲突，创建一个新的 Conda 环境，使用 Python 3.10（`gymnasium` 与 Python 3.11 可能有兼容性问题）：

```bash
conda create -n RL_gymnasium python=3.10
conda activate RL_gymnasium
```

**验证 Python 版本**：

```bash
python --version
```

**预期输出**：`Python 3.10.x`

### 第二步：安装 SWIG

`box2d-py` 包需要 `swig` 进行编译，用于支持 Box2D 环境（如 `LunarLander-v3`）。通过 Conda 安装 `swig`：

```bash
conda install swig
```

**验证 SWIG 安装**：

```bash
swig -version
```

**预期输出**：显示 SWIG 版本（如 `SWIG Version 4.0.2`）。

### 第三步：安装 Gymnasium 及依赖

安装 `gymnasium` 及其所有可选依赖（包括 Atari、Box2D、MuJoCo 等）：

```bash
pip install gymnasium[all] --no-cache-dir -i https://pypi.org/simple
```

* `--no-cache-dir`：避免缓存导致的依赖冲突。
* `-i https://pypi.org/simple`：使用默认 PyPI 源，防止镜像（如阿里云）缺少包。

此命令安装以下关键依赖：

* `gymnasium==1.1.1`：强化学习环境库
* `ale_py==0.11.0`：Atari 环境支持
* `box2d-py==2.3.5`：Box2D 环境支持
* `pygame==2.6.1`：环境渲染
* `numpy==2.2.5`：数值计算
* `torch==2.7.0`：深度学习框架（用于 DQN 和 PPO）

**注意**：

* 如果遇到 `error: command 'swig.exe' failed`，说明 `swig` 未安装，重新运行第二步。
* 如果安装失败，检查网络连接，确保 PyPI 源可用。

### 第四步：安装 Atari ROM 文件

Atari 环境（如 `PongNoFrameskip-v4`）需要 ROM 文件（如 `pong.bin`），通过 `autorom` 包管理。由于 `gymnasium==1.1.1` 不支持 `accept-rom-license` 额外选项，需单独安装 `autorom`：

```bash
pip install autorom[accept-rom-license] --no-cache-dir -i https://pypi.org/simple
```

此命令自动下载 ROM 文件到默认路径：

```
C:\Users\<你的用户名>\.conda\envs\RL_gymnasium\Lib\site-packages\auto_rom\roms
```

#### 验证 ROM 文件

检查 ROM 文件是否存在：

```bash
dir C:\Users\<你的用户名>\.conda\envs\RL_gymnasium\Lib\site-packages\auto_rom\roms
```

**预期输出**：列出 `pong.bin`、`breakout.bin` 等文件。

**如果目录为空**：

1. 强制重新安装 `autorom`：

   ```bash
   pip install autorom[accept-rom-license] --force-reinstall -i https://pypi.org/simple
   ```
2. 手动触发 ROM 下载：

   ```python
   from AutoROM.accept_rom_license import install_roms
   install_roms()
   ```

   保存为 `install_roms.py`，然后运行：

   ```bash
   python install_roms.py
   ```
3. 再次检查 ROM 文件：

   ```bash
   dir C:\Users\<你的用户名>\.conda\envs\RL_gymnasium\Lib\site-packages\auto_rom\roms
   ```

#### 验证 ROM 注册

使用 `ale_py==0.11.0` 的 API 检查注册的 ROM：

```bash
python -c "from ale_py.roms import get_all_rom_ids; print(get_all_rom_ids())"
```

**预期输出**：`['pong', 'breakout', 'space_invaders', ...]`（包含 `pong`）。

**如果输出为空**：

* ROM 文件未被 `ale_py` 识别，尝试手动导入（需合法 ROM 文件）：
  1. 将 ROM 文件（如 `pong.bin`）放入本地目录（如 `C:\roms`）。
  2. 导入 ROM：

     ```bash
     ale-import-roms C:\roms
     ```
  3. 再次验证：

     ```bash
     python -c "from ale_py.roms import get_all_rom_ids; print(get_all_rom_ids())"
     ```

### 第五步：验证依赖

检查关键依赖是否正确安装：

```bash
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import ale_py; print(ale_py.__version__)"
python -c "import pygame; print(pygame.__version__)"
python -c "import Box2D; print('Box2D OK')"
python -c "import torch; print(torch.__version__)"
```

**预期输出**：

* `1.1.1`（gymnasium）
* `0.11.0`（ale\_py）
* `2.6.1`（pygame）
* `Box2D OK`
* `2.7.0`（torch）

**如果缺少依赖**：

```bash
pip install gymnasium[all] torch --no-cache-dir -i https://pypi.org/simple
```

### 第六步：测试环境

在运行强化学习算法前，测试 Atari 和 Box2D 环境是否正常。

#### 测试 Atari 环境

验证 `PongNoFrameskip-v4`：

```python
import gymnasium as gym
import ale_py #不能少
env = gym.make("PongNoFrameskip-v4", render_mode="human")
state, _ = env.reset()
env.render()
env.close()
```

**预期结果**：Pong 游戏窗口显示。

#### 测试 Box2D 环境

验证 `LunarLander-v3`：

```python
import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="human")
state, _ = env.reset()
env.render()
env.close()
```

**预期结果**：LunarLander 窗口显示。

**如果测试失败**，参考“故障排查”部分。

### 第七步：图像预处理和帧堆叠

Atari 环境的原始观察是 RGB 图像（210x160x3），需要预处理（如灰度化、裁剪、缩放）和帧堆叠（通常堆叠 4 帧）以提高性能。以下是预处理和帧堆叠的实现：

```python
import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class AtariPreprocessor:
    def __init__(self, env, frame_stack=4, img_size=(84, 84)):
        self.env = env
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.stack = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        # 转换为灰度
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 裁剪无关区域（Pong 的得分区域）
        frame = frame[34:194, :]  # 裁剪到 160x160
        # 缩放到 84x84
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        # 归一化到 [0, 1]
        frame = frame / 255.0
        return frame

    def reset(self):
        state, _ = self.env.reset()
        frame = self.preprocess_frame(state)
        # 初始化帧堆叠
        for _ in range(self.frame_stack):
            self.stack.append(frame)
        return np.stack(self.stack, axis=0)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        frame = self.preprocess_frame(next_state)
        self.stack.append(frame)
        return np.stack(self.stack, axis=0), reward, terminated, truncated, info

# 测试预处理
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
preprocessor = AtariPreprocessor(env)
state = preprocessor.reset()
print("Preprocessed state shape:", state.shape)  # 应为 (4, 84, 84)
env.close()
```

保存为 `atari_preprocessor.py`，用于后续 DQN 和 PPO 训练。

**说明**：

* **灰度化**：减少计算量，保留关键信息。
* **裁剪**：移除得分区域，聚焦游戏画面。
* **缩放**：84x84 是 Atari 强化学习的标准尺寸。
* **帧堆叠**：堆叠 4 帧捕捉运动信息。

### 第八步：实现 DQN 算法

以下是 DQN 实现，结合帧堆叠预处理，针对 `PongNoFrameskip-v4`。

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from atari_preprocessor import AtariPreprocessor

# DQN 网络
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# DQN 智能体
class DQNAgent:
    def __init__(self, env, preprocessor, device="cpu"):
        self.env = env
        self.preprocessor = preprocessor
        self.device = device
        self.q_network = DQN((4, 84, 84), env.action_space.n).to(device)
        self.target_network = DQN((4, 84, 84), env.action_space.n).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.epsilon = 1.0
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.target_update_freq = 1000
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated or truncated))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# 训练循环
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
preprocessor = AtariPreprocessor(env)
agent = DQNAgent(env, preprocessor)
episodes = 1000

for episode in range(episodes):
    state = preprocessor.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = preprocessor.step(action)
        agent.store_transition(state, action, reward, next_state, terminated, truncated)
        agent.train()
        state = next_state
        total_reward += reward
        done = terminated or truncated
        if done:
            agent.epsilon = max(0.1, agent.epsilon * 0.995)
            print(f"回合 {episode+1}, 总奖励: {total_reward}")
env.close()
```

保存为 `dqn_pong.py`，运行：

```bash
python dqn_pong.py
```

**说明**：

* **帧堆叠**：输入为 (4, 84, 84)，通过 `AtariPreprocessor` 实现。
* **目标网络**：每 1000 步更新，稳定训练。
* **探索率**：`epsilon` 从 1.0 衰减到 0.1。

### 第九步：实现 PPO 算法

以下是 PPO 实现，同样结合帧堆叠，针对 `PongNoFrameskip-v4`。

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from atari_preprocessor import AtariPreprocessor

# Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        action_probs = self.actor(conv_out)
        value = self.critic(conv_out)
        return action_probs, value

# PPO 智能体
class PPOAgent:
    def __init__(self, env, preprocessor, device="cpu"):
        self.env = env
        self.preprocessor = preprocessor
        self.device = device
        self.policy = ActorCritic((4, 84, 84), env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2.5e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.memory.append((state, action, reward, log_prob, value, done))

    def train(self):
        states, actions, rewards, log_probs, values, dones = zip(*self.memory)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算回报
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum * (1 - done)
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(self.device)

        # 计算优势
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 更新
        for _ in range(10):
            action_probs, new_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

# 训练循环
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
preprocessor = AtariPreprocessor(env)
agent = PPOAgent(env, preprocessor)
episodes = 1000

for episode in range(episodes):
    state = preprocessor.reset()
    total_reward = 0
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = preprocessor.step(action)
        _, value = agent.policy(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
        done = terminated or truncated
        agent.store_transition(state, action, reward, log_prob, value, done)
        state = next_state
        total_reward += reward
        if done:
            agent.train()
            print(f"回合 {episode+1}, 总奖励: {total_reward}")
env.close()
```

保存为 `ppo_pong.py`，运行：

```bash
python ppo_pong.py
```

**说明**：

* **帧堆叠**：与 DQN 一致，使用 4 帧堆叠。
* **PPO 损失**：包含剪切比率、价值损失和熵正则化。
* **超参数**：`lr=2.5e-4`、`eps_clip=0.2` 适合 Pong，需根据实验调整。

### 第十步：保存训练视频

为可视化训练效果，可保存环境渲染的视频：

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from atari_preprocessor import AtariPreprocessor

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
env = RecordVideo(env, video_folder="pong_videos", episode_trigger=lambda x: x % 100 == 0)
preprocessor = AtariPreprocessor(env)
state = preprocessor.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    next_state, reward, terminated, truncated, _ = preprocessor.step(action)
    state = next_state
    if terminated or truncated:
        state = preprocessor.reset()
env.close()
```

**说明**：

* **视频保存**：每 100 回合保存一次视频，存储在 `pong_videos` 目录。
* **依赖**：需要 `moviepy` 包，自动由 `gymnasium[all]` 安装。
* **使用**：可替换随机动作为 DQN 或 PPO 智能体的动作。

### 故障排查

以下针对常见报错和常见问题提供详细解决方案。

#### 1\. `AttributeError: module 'ale_py.roms' has no attribute '__all__'`

* **原因**：`ale_py==0.11.0` 不使用 `__all__`，教程中旧命令不适用。
* **解决**：使用新 API：

  ```bash
  python -c "from ale_py.roms import get_all_rom_ids; print(get_all_rom_ids())"
  ```

#### 2\. `ImportError: cannot import name 'list_roms' from 'ale_py.roms'`

* **原因**：`list_roms` 在 `ale_py==0.11.0` 中不存在。
* **解决**：同上，使用 `get_all_rom_ids()`。

#### 3\. `get_all_rom_ids()` 返回空列表

* **原因**：ROM 文件未下载或未注册。
* **解决**：
  1. 检查 ROM 文件：

     ```bash
     dir C:\Users\<你的用户名>\.conda\envs\RL_gymnasium\Lib\site-packages\auto_rom\roms
     ```
  2. 重新安装 `autorom`：

     ```bash
     pip install autorom[accept-rom-license] --force-reinstall -i https://pypi.org/simple
     ```
  3. 手动触发 ROM 下载：

     ```bash
     python install_roms.py
     ```
  4. 手动导入 ROM（需合法文件）：

     ```bash
     ale-import-roms C:\roms
     ```

#### 4\. `error: command 'swig.exe' failed`

* **原因**：缺少 `swig`，导致 `box2d-py` 编译失败。
* **解决**：

  ```bash
  conda install swig
  pip install gymnasium[all] --no-cache-dir
  ```

#### 5\. `NameNotFound: Environment 'Pong' doesn't exist`

* **原因**：使用了错误的 ID（`Pong-v0`），`gymnasium` 需要 `PongNoFrameskip-v4`。
* **解决**：
  * 修改代码：

    ```python
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    ```
  * 验证 ROM 注册（见问题 3）。

#### 6\. `Error: We're Unable to find the game "Pong"`

* **原因**：ROM 文件未注册。
* **解决**：同问题 3，重新安装或导入 ROM。

#### 7\. 网络或镜像问题

* **原因**：阿里云镜像（`https://mirrors.aliyun.com/pypi/simple`）可能缺少包。
* **解决**：使用默认 PyPI：

  ```bash
  pip install <package> -i https://pypi.org/simple
  ```

#### 8\. 依赖冲突或环境损坏

* **解决**：重建环境：

  ```bash
  conda create -n gymnasium_new python=3.10
  conda activate gymnasium_new
  conda install swig
  pip install gymnasium[all] autorom[accept-rom-license] --no-cache-dir
  ```

### 其他实用功能

1. **保存训练模型**：在 DQN 或 PPO 训练后保存模型：

   ```python
   torch.save(agent.q_network.state_dict(), "dqn_pong.pth")  # DQN
   torch.save(agent.policy.state_dict(), "ppo_pong.pth")    # PPO
   ```
2. **加载模型**：加载保存的模型继续训练或测试：

   ```python
   agent.q_network.load_state_dict(torch.load("dqn_pong.pth"))  # DQN
   agent.policy.load_state_dict(torch.load("ppo_pong.pth"))     # PPO
   ```
3. **绘制奖励曲线**：
   使用 Matplotlib 记录和可视化奖励：

   ```python
   import matplotlib.pyplot as plt

   rewards = []
   for episode in range(episodes):
       # ... 训练代码 ...
       rewards.append(total_reward)
       if episode % 10 == 0:
           plt.plot(rewards)
           plt.xlabel("回合")
           plt.ylabel("总奖励")
           plt.savefig("reward_curve.png")
           plt.clf()
   ```

### 资源与支持

* **官方文档**：
  * [Gymnasium](https://gymnasium.farama.org/)
  * [AutoROM](https://github.com/Farama-Foundation/AutoROM)
  * [ALE-Py](https://github.com/mgbellemare/Arcade-Learning-Environment)
* **论文**：
  * [DQN](https://arxiv.org/abs/1312.5602)
  * [PPO](https://arxiv.org/abs/1707.06347)
* **问题反馈**：如果遇到问题，提供以下信息：
  * `python -c "from ale_py.roms import get_all_rom_ids; print(get_all_rom_ids())"` 的输出
  * `dir C:\Users\<你的用户名>\.conda\envs\RL_gymnasium\Lib\site-packages\auto_rom\roms` 的输出
  * 运行 DQN/PPO 代码或测试代码的完整错误日志

### 注意事项

* **训练时间**：Pong 的 DQN 和 PPO 训练可能需要数小时到数天，建议监控奖励变化。
* **超参数调整**：根据实验结果调整学习率、折扣因子等。
* **GPU 加速**：如有 GPU，设置 `device="cuda"` 加速训练。
* **合法 ROM**：确保 ROM 文件来源合法，避免法律风险。


### ref

https://blog.csdn.net/Rhett_Butler0922/article/details/147721024
