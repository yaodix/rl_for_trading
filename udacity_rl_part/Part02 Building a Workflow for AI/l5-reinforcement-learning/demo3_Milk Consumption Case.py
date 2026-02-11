'''
，这个代码是在用强化学习解决一个控制问题（控制牛奶消耗量），通过控制结果来间接实现分类效果。
'''

import sys
import os
from pathlib import Path


# Now you can import func_lib
import func_lib
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium import spaces
import matplotlib.pyplot as plt

# Generate the training dataset
np.random.seed(42)
days = 100
milk_consumption = np.random.uniform(0, 10, size=days)  # Random milk consumption between 0 and 10 ounces
discomfort = (milk_consumption > 5).astype(int)  # Discomfort turns to 1 if consumption > 5 ounces

# Create a DataFrame for the training dataset
data = pd.DataFrame({
    'day': range(1, days + 1),
    'milk_consumption': milk_consumption,
    'discomfort': discomfort
})

# Generate the testing dataset
np.random.seed(24)
days_test = 50
milk_consumption_test = np.random.uniform(0, 10, size=days_test)  # Random milk consumption between 0 and 10 ounces
discomfort_test = (milk_consumption_test > 5).astype(int)  # Discomfort turns to 1 if consumption > 5 ounces

# Create a DataFrame for the testing dataset
data_test = pd.DataFrame({
    'day': range(1, days_test + 1),
    'milk_consumption': milk_consumption_test,
    'discomfort': discomfort_test
})

# Define the custom environment for milk consumption
class MilkConsumptionEnv(gym.Env):
    def __init__(self, data):
        super(MilkConsumptionEnv, self).__init__()
        self.data = data
        self.current_day = 0
        self.action_space = spaces.Discrete(2)  # 0: consume less milk, 1: consume more milk
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        obs = np.array([self.data.iloc[self.current_day]['milk_consumption']], dtype=np.float32)
        return obs, {}
    def step(self, action):
      self.current_day += 1
      
      if action == 0:  # consume less milk
          next_obs = max(0, self.data.iloc[self.current_day]['milk_consumption'] - 1)
      else:  # consume more milk
          next_obs = min(10, self.data.iloc[self.current_day]['milk_consumption'] + 1)
      
      next_obs = np.array([next_obs], dtype=np.float32)
      done = self.current_day >= len(self.data) - 1
      
      # Reward is -1 if discomfort (1), 0 otherwise, 根据明天是否不适得到奖励
      reward = -1 if self.data.iloc[self.current_day]['discomfort'] == 1 else 0
      
      return next_obs, reward, done, False, {}
    
if __name__ == '__main__':
    env = MilkConsumptionEnv(data)
    agent = DQN('MlpPolicy', env, verbose=0)
    print('Start training...')
    agent.learn(total_timesteps=50000)
    
    print('Training completed.')
    
    env_test = MilkConsumptionEnv(data_test)
    correct_predictions = 0
    total_predictions = 0
    
    obs, _ = env_test.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs)  # 更新策略，未来做出更好决策
        obs, reward, done, _, _ = env_test.step(action)

        # The agent's action is based on whether it predicts discomfort or not
        predicted_discomfort = 1 if obs[0] > 5 else 0

        # Compare prediction with actual discomfort
        actual_discomfort = env_test.data.iloc[env_test.current_day - 1]['discomfort']
        if predicted_discomfort == actual_discomfort:
            correct_predictions += 1
        total_predictions += 1
    acc = correct_predictions / total_predictions * 100
    print(f"Testing Accuracy: {acc:.2f}%")

  
  