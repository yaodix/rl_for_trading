import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium.wrappers import HumanRendering
import time

# Step 1: 创建环境并使用HumanRendering包装器
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = HumanRendering(env)  # 这会自动处理渲染

# Step 2: 创建DQN智能体
agent = DQN('MlpPolicy', env, verbose=0)

# Step 3: 训练智能体
print("开始训练...")
agent.learn(total_timesteps=10000)
print("训练完成!")

# Step 4: 测试智能体
obs, _ = env.reset()
done = False
truncated = False

for episode in range(5):  # 测试5个回合
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not done and not truncated:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # 添加延迟以便观察
        time.sleep(0.05)
    
    print(f"回合 {episode + 1}: 总奖励 = {total_reward}")

env.close()