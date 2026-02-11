import gymnasium as gym
from stable_baselines3 import DQN
import time

def train_and_save():
    """训练并保存DQN模型"""
    
    # Step 1: 创建训练环境（无渲染以加速）
    print("创建训练环境...")
    env = gym.make('CartPole-v1')
    
    # Step 2: 创建DQN智能体
    print("初始化DQN智能体...")
    agent = DQN(
        'MlpPolicy', 
        env, 
        verbose=1,          # 显示训练进度
        learning_rate=0.001,
        buffer_size=10000,  # 经验回放缓冲区大小
        learning_starts=1000,  # 开始学习前先收集的经验数
        batch_size=64,      # 训练批次大小
        tau=1.0,            # 目标网络更新率
        gamma=0.99,         # 折扣因子
        train_freq=4,       # 每4步训练一次
        gradient_steps=1,
        target_update_interval=100,  # 每100步更新目标网络
        exploration_fraction=0.1,    # 探索率衰减时间比例
        exploration_initial_eps=1.0, # 初始探索率
        exploration_final_eps=0.05,  # 最终探索率
        tensorboard_log="./dqn_cartpole_tensorboard/"  # TensorBoard日志
    )
    
    # Step 3: 训练智能体
    print("开始训练...")
    start_time = time.time()
    agent.learn(total_timesteps=50000)  # ✅ 足够的训练步数
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # Step 4: 保存模型
    print("保存模型...")
    save_path = "./dqn_cartpole_model"
    agent.save(save_path)
    print(f"模型已保存到: {save_path}")
    
    # Step 5: 关闭环境
    env.close()
    
    return agent, save_path

def test_model(model_path):
    """测试已保存的模型"""
    
    print("\n测试模型...")
    
    # 创建测试环境（带渲染）
    env = gym.make('CartPole-v1', render_mode='human')
    
    # 加载模型
    print(f"从 {model_path} 加载模型...")
    agent = DQN.load(model_path, env=env)
    
    # 运行测试episode
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    print("开始测试运行...")
    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # 可选：减慢渲染速度以便观察
        time.sleep(0.1)
    
    print(f"测试完成，总奖励: {total_reward}")
    env.close()

if __name__ == "__main__":
    # 训练并保存
    model_path = "./dqn_cartpole_model"
    # agent, model_path = train_and_save()
    
    # 测试模型
    test_model(model_path)