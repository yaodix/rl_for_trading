'''
运行这个程序可以看到Q-learning如何学习预测金融收益的方向（正收益或负收益）
同Exercise1_solution.ipynb
'''

import sys
import os
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import defaultdict
from gymnasium import spaces

class ReturnEnv(gym.Env):
    def __init__(self, df):
        super(ReturnEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        # Initialize dataframe
        self.df = df
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        state = np.array([self.df.iloc[self.current_step]['1_d_returns']], dtype=np.float32)
        return state, {}
        
    def step(self, action):
        target = self.df.iloc[self.current_step]['Target_Returns']
        
        # Reward if action matches target return
        reward = 1.0 if action == target else -1.0
        
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        if not done:
            # next_state = np.array([self.df.iloc[self.current_step]['1_d_returns']], dtype=np.float32)
            next_state = self.df.iloc[self.current_step][['1_d_returns']].values
        else:
            next_state = np.zeros(1, dtype=np.float32)
        
        return next_state, reward, done, False, {}

class QLearningAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # explore
        else:
            state_key = str(state.tolist()) if isinstance(state, np.ndarray) else str(state)
            return np.argmax(self.q_table[state_key])  # exploit
            
    def update(self, state, action, reward, next_state):
        state_key = str(state.tolist()) if isinstance(state, np.ndarray) else str(state)
        next_state_key = str(next_state.tolist()) if isinstance(next_state, np.ndarray) else str(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

def main():
    # Example data
    df = pd.DataFrame({
        'Target_Returns': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        '1_d_returns': [0.062030, -0.038076, 0.050, 0.030, -0.020, 0.062030, -0.038076, 0.050, 0.0330, -0.020]
    })
    
    print("数据信息:")
    print(f"数据行数: {len(df)}")
    print(f"正收益天数: {sum(df['Target_Returns'] == 1)}")
    print(f"负收益天数: {sum(df['Target_Returns'] == 0)}")

    # Create environment and agent
    env = ReturnEnv(df)
    agent = QLearningAgent(env.action_space, env.observation_space)

    # Training loop
    n_episodes = 1000
    print(f"\n开始训练，共 {n_episodes} 轮...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        # 每个episode的统计
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            
            total_reward += reward
            steps += 1
        
        # 每100轮打印一次训练进度
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: 总奖励 = {total_reward:.2f}, 步数 = {steps}")

    print("\n训练完成。")

    # Evaluation
    print("\n" + "="*50)
    print("开始评估...")
    print("="*50)
    
    state, _ = env.reset()
    done = False
    step = 1
    
    # 评估时的统计
    total_test_reward = 0
    correct_predictions = 0
    total_predictions = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        # 获取实际目标
        actual_target = env.df['Target_Returns'].values[step-1]
        
        # 统计预测准确率
        if action == actual_target:
            correct_predictions += 1
        total_predictions += 1
        
        # Print detailed step information
        print(f"Step {step}: 动作 = {action}, 目标收益 = {actual_target}, " 
              f"奖励 = {reward}, 状态 = {state[0]:.4f}, {'正确' if action == actual_target else '错误'}")
        print(f"Action: {action}, Target Return: {env.df['Target_Returns'].values[step-1]}, Reward: {reward}, Step: {step}, State: {state}, Next State: {next_state}")
        state = next_state
        total_test_reward += reward
        step += 1
    
    # 输出评估结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"总奖励: {total_test_reward}")
    print(f"预测准确率: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions*100:.2f}%")
    
    # 输出Q表统计信息
    print(f"\nQ表大小: {len(agent.q_table)} 个状态")
    # 输出Q表示例
    print("\nQ表示例:")
    for state, actions in list(agent.q_table.items()):
        print(f"状态 {state}: {actions}")


if __name__ == "__main__":
    main()