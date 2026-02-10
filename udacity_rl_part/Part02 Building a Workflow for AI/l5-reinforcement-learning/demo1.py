import sys
import os
  
import gymnasium as gym
from stable_baselines3 import DQN
import time
import pygame

# Step 1: 创建环境，设置渲染模式为 'human'（可视化）
env = gym.make('CartPole-v1', render_mode='human')

# Step 2: 创建 DQN 智能体，关闭详细日志（verbose=0）
agent = DQN('MlpPolicy', env, verbose=0)

# Step 3: 训练智能体 —— 学习 10,000 步
agent.learn(total_timesteps=1000)

print("Training completed.")

# Step 4: 测试训练好的智能体
obs, _ = env.reset()
done = False

try:
    while not done:
        # 根据当前观测状态预测动作
        action, _ = agent.predict(obs)
        
        # 执行动作，获得新状态、奖励、是否结束等
        obs, reward, done, _, _ = env.step(action)
        
        # 渲染环境（可视化显示小车摆杆）
        env.render()
        
        # 添加微小延迟，以便人眼观察动画
        time.sleep(0.05)

except SystemExit:
    # 用户主动关闭窗口时退出 Pygame
    pygame.quit()