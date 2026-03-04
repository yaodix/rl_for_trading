import numpy as np
import torch
from lib import environ
from lib import config

METRICS = (
    'episode_reward',
    'episode_steps',
    'order_profits',
    'order_steps',
)

def safe_mean(vals):
    """安全计算均值，避免空列表报错"""
    return np.mean(vals) if len(vals) > 0 else 0.0

def validation_run(env, net, episodes=100, device="cpu", epsilon=0.0, 
                   commission=0.5/10000, slippage=0.0001):
    """
    验证函数（修复版）
    commission: 佣金率（万分比，如 0.5/10000）
    slippage: 滑点率（如 0.0001 表示万 1）
    """
    stats = { metric: [] for metric in METRICS }
    
    # 佣金最少5元，根据cash计算调整后佣金
    comm = 5.0 / config.env_config["cash"]
    commission = max(comm, commission)

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        episode_steps = 0
        
        # 用环境的状态跟踪持仓，避免重复逻辑
        position_open_price = None
        position_start_step = None

        while True:
            obs_v = torch.tensor([obs], dtype=torch.float32).to(device)
            with torch.no_grad():  # 👈 验证时不需要梯度
                out_v = net(obs_v)
            
            action_idx = out_v.max(dim=1)[1].item()
            # 验证时强制贪婪（epsilon 应为 0）
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            
            close_price = env._state._cur_close()
            action = environ.Actions(action_idx)

            # 记录开仓信息（用于计算单笔收益）
            if action == environ.Actions.Buy and position_open_price is None:
                exec_price = close_price * (1 + slippage)  # 👈 买入考虑滑点
                position_open_price = exec_price
                position_start_step = episode_steps
            
            # 计算平仓收益
            elif action == environ.Actions.Close and position_open_price is not None:
                exec_price = close_price * (1 - slippage)  # 👈 卖出考虑滑点
                # 修复：佣金计算去掉多余的 /100
                profit = exec_price - position_open_price - (exec_price + position_open_price) * commission
                profit_pct = 100.0 * profit / position_open_price  # 百分比收益
                stats['order_profits'].append(profit_pct)
                stats['order_steps'].append(episode_steps - position_start_step)
                position_open_price = None
                position_start_step = None

            obs, reward, done, _, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            
            if done:
                # Episode 结束时如果还有持仓，按市价强制平仓计算收益
                if position_open_price is not None:
                    exec_price = close_price * (1 - slippage)
                    profit = exec_price - position_open_price - (exec_price + position_open_price) * commission
                    profit_pct = 100.0 * profit / position_open_price
                    stats['order_profits'].append(profit_pct)  # %order_profit
                    stats['order_steps'].append(episode_steps - position_start_step)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    # 修复：用 safe_mean 避免空列表报错
    return { key: safe_mean(vals) for key, vals in stats.items() }