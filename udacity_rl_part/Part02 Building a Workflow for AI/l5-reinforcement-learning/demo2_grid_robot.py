import sys
import os

import random
import numpy as np
import gymnasium as gym
from collections import defaultdict
import time

class SimpleGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5):
        super(SimpleGridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = gym.spaces.MultiDiscrete([grid_size, grid_size])
        self.state = None
        self.goal = (grid_size - 1, grid_size - 1)

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state, dtype=np.int32), {}  # gymnasium 需要返回 info 字典

    def step(self, action):
        x, y = self.state

        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)

        self.state = (x, y)

        done = self.state == self.goal
        reward = 1 if done else -0.1

        return np.array(self.state, dtype=np.int32), reward, done, False, {}
    
    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.goal] = 'G'
        x, y = self.state
        grid[x, y] = 'A'
        print("\n".join("".join(row) for row in grid))
        print()

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            # 确保state是tuple类型
            state_tuple = tuple(state) if isinstance(state, np.ndarray) else state
            return np.argmax(self.q_table[state_tuple])

    def learn(self, state, action, reward, next_state):
        # 确保state和next_state都是tuple类型
        state_tuple = tuple(state) if isinstance(state, np.ndarray) else state
        next_state_tuple = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        
        best_next_action = np.argmax(self.q_table[next_state_tuple])
        td_target = reward + self.discount_factor * self.q_table[next_state_tuple][best_next_action]
        td_error = td_target - self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] += self.learning_rate * td_error

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        state = tuple(state)  # 转换为tuple
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = tuple(next_state)  # 转换为tuple
            agent.learn(state, action, reward, next_state)
            state = next_state
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed")

if __name__ == "__main__":
    # 1. 创建环境和智能体
    env = SimpleGridEnv()
    agent = QLearningAgent(env)
    
    # 2. 训练阶段
    print("开始训练...")
    train_agent(env, agent, episodes=1000)
    print("训练完成！")
    
    # 3. 测试/演示阶段
    print("开始演示...")
    state, _ = env.reset()
    state = tuple(state)  # 转换为tuple
    done = False
    total_reward = 0
    
    while not done:
        # 渲染当前状态
        env.render()
        
        # 智能体选择动作（训练完成后使用贪婪策略）
        action = agent.choose_action(state, epsilon=0)  # 无探索
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        next_state = tuple(next_state)
        
        # 更新
        state = next_state
        total_reward += reward
        time.sleep(1.5)
    
    print(f"测试结束，总奖励: {total_reward}")
    env.close()