'''
This method is useful for applying reinforcement learning to financial markets, 
aiming to develop predictive trading algorithms.
'''

import sys
import os
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from gymnasium import spaces
import matplotlib.pyplot as plt

# # Manually set the path relative to the py file's location that you want to import
# func_lib_path = os.path.abspath(os.path.join(os.getcwd(), '../'))  # Add the path to sys.path
# sys.path.append(func_lib_path)

# Now you can import func_lib
import func_lib

# ============ 1. 数据准备 ============
# 创建历史价格数据
# historical_prices = func_lib.createHistPrices()
hist_file_path = '/home/yao/myproject/rl_for_trading/udacity_rl_part/Part02 Building a Workflow for AI/historical_prices.csv'
historical_prices = func_lib.load_historical_prices(hist_file_path)

# 设置截止日期，过滤数据
cutoff_date = pd.to_datetime('2022-10-10')
historical_prices = historical_prices[historical_prices.index.get_level_values('Date') > cutoff_date]
list_of_momentums = [1, 5, 15, 20]
total_returns = func_lib.compute_returns(historical_prices, list_of_momentums)
total_returns.dropna(inplace=True)

# Converting the 'F_1_d_returns' to binary based on whether the value is positive or not
total_returns['F_1_d_returns_Ind'] = total_returns['F_1_d_returns'].apply(lambda x: 1 if x > 0 else 0)
total_returns.head()

# Determine the split index for 70% of the dates
unique_dates = total_returns.index.get_level_values('Date').unique()
split_date = unique_dates[int(0.7 * len(unique_dates))]
split_date

# Create the training set: all data before the split date
train_data = total_returns.loc[total_returns.index.get_level_values('Date') < split_date]

# Create the testing set: all data from the split date onwards
test_data = total_returns.loc[total_returns.index.get_level_values('Date') >= split_date]

total_returns = test_data['F_1_d_returns']

features = ['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']
target = ['F_1_d_returns_Ind']

# Split the data into training and testing sets
X_train = train_data[features]
X_test = test_data[features]
y_train = train_data[target]
y_test = test_data[target]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# Define the custom gym environment
class ReturnEnv(gym.Env):
    def __init__(self, df):
        super(ReturnEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize dataframe
        self.df = df.reset_index()
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        state = self.df.iloc[self.current_step][['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']].values
        return state

    def step(self, action):
        target = self.df.iloc[self.current_step]['F_1_d_returns_Ind']

        # Reward if action matches target return
        reward = 1 if action == target else -1

        self.current_step += 1
        done = self.current_step >= len(self.df)

        if not done:
            next_state = self.df.iloc[self.current_step][['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']].values
        else:
            next_state = np.zeros(4)

        return next_state, reward, done, {}

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # explore
        else:
            return np.argmax(self.q_table[str(state)])  # exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[str(next_state)])
        td_target = reward + self.gamma * self.q_table[str(next_state)][best_next_action]
        td_error = td_target - self.q_table[str(state)][action]
        self.q_table[str(state)][action] += self.alpha * td_error

def trading_strategy(y_pred):
    if y_pred > 0.5:
        return 1  # Go Long
    else:
        return 0
      
def print_predictions(env, agent, df, dataset_type=""):
    """
    对数据集进行预测并打印结果
    
    Args:
        env: 强化学习环境
        agent: 训练好的智能体
        df: 需要预测的数据框
        dataset_type: 数据集类型（training/evaluation）
    
    Returns:
        添加了预测列的数据框
    """
    state = env.reset()
    done = False
    step = 0
    predictions = []
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        predictions.append(action)
        state = next_state
        step += 1
    
    df_copy = df.copy()
    df_copy['Prediction'] = predictions
    
    correct_predictions = (df_copy['Prediction'] == df_copy['F_1_d_returns_Ind']).sum()
    total_predictions = len(df_copy)
    accuracy = correct_predictions / total_predictions
    
    print(f"\n{dataset_type} Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    return df_copy

def main():
    train_df = pd.merge(y_train, X_train_scaled, left_index=True, right_index=True)
    test_df = pd.merge(y_test, X_test_scaled, left_index=True, right_index=True)

    train_df.index.names = ['Ticker', 'Date']
    test_df.index.names = ['Ticker', 'Date']

    # Create environment and agent for training
    env = ReturnEnv(train_df)
    agent = QLearningAgent(env.action_space, env.observation_space)

    # Training Loop
    n_episodes = 10
    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    print("Training finished.")

    # Print predictions for training data and add them back to df
    train_df = print_predictions(env, agent, train_df, "training")

    # Create environment for evaluation
    eval_env = ReturnEnv(test_df)

    # Print predictions for evaluation data and add them back to test_df
    test_df = print_predictions(eval_env, agent, test_df, "evaluation")

    print("\nTraining Data with Predictions:")
    print(train_df)

    print("\nEvaluation Data with Predictions:")
    print(test_df)
    
    # train_df.to_csv('train.csv')
    # test_df.to_csv('test.csv')

if __name__ == "__main__":
    main()
