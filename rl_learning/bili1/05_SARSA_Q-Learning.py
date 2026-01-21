'''
蒙特卡洛（Monte Carlo, MC）方法 可以 从环境中获得即时奖励（每一步的 ），但它 
只在一幕结束之后 才用这些奖励来计算总回报 ，并更新价值函数。

时序差分算法可以在每一步都更新价值函数，而不需要等待一幕结束。
'''

from help import FrozenLake, print_policy, test_game
import numpy as np

def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values
  
alphas = decay_schedule(1, 0.0001, 0.8, 20000)

import matplotlib.pylab as plt

# plt.plot(alphas)
# plt.savefig('alphas.png')
def select_action(state, Q, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(len(Q[state]))
      

def sarsa(env, episodes=100, gamma=0.9, test_policy_freq=1000):
    nS, nA = 16, 4
    Q = np.zeros((nS, nA), dtype=np.float64)
    alphas = decay_schedule(0.5,0.01,0.5, episodes)
    epsilons = decay_schedule(1,0.01,0.8, episodes)
    
    for i in range(episodes): 
        state = env.reset()
        finished = False
        action = select_action(state, Q, epsilons[i])
        while not finished:
            next_state, reward, finished = env.step(action)
            next_action = select_action(next_state, Q, epsilons[i])
            target = reward + gamma * Q[next_state][next_action] * (not finished)
            error = target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[i] * error
            state, action = next_state, next_action

        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
        if i % test_policy_freq == 0:
                print("Test episode {} Reaches goal {:.2f}%. ".format
                (i, test_game(env, pi,)*100))

    return pi, Q
  
def q_learning(env,episodes=100,gamma=0.9,test_policy_freq=1000):
    nS, nA = 16,4
    Q = np.zeros((nS, nA), dtype=np.float64)
    alphas = decay_schedule(0.5,0.01,0.5, episodes)
    epsilons = decay_schedule(1,0.01,0.8, episodes)
    for i in range(episodes): 
        state = env.reset()
        finished = False
        while not finished:
            action = select_action(state, Q, epsilons[i])
            next_state, reward, finished = env.step(action)
            # 学习所用的Q值和sarsa不一样
            target = reward + gamma * Q[next_state].max() * (not finished)
            # 计算目标值（基于贝尔曼最优方程）
            error = target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[i] * error
            state = next_state


        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
        if i % test_policy_freq == 0:
                print("Test episode {} Reaches goal {:.2f}%. ".format
                (i, test_game(env, pi)*100))

    return pi, Q
  

if __name__ == '__main__':
  env = FrozenLake()
  policy_sarsa, Q_sarsa = sarsa(env, episodes=20000)  # sarsa所利用的信息中噪声较少，学习比蒙特卡罗更快
  print_policy(policy_sarsa)
  
  policy_qlearning,Q_qlearning = q_learning(env, episodes=20000)
  print_policy(policy_qlearning)