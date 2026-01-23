'''
Q表缺点
1.无法处理state/action过多的问题
2.无法出连续的state/action问题
3.不具备泛化能力

'''

import torch
from torch.nn import Linear
import numpy as np
import gymnasium as gym

def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values
  
def one_hot(x, size):
    result = np.zeros(size)
    result[x] = 1
    return result 
  
def conv2tensor(x,size):
    x = one_hot(x,size)
    x = torch.from_numpy(x).float()
    return x
  
  
def select_action(q_value, epsilon):
    q_value_np = q_value.clone().detach().numpy().squeeze()
    if np.random.random() > epsilon:
        final_move = q_value_np.argmax()
    else:
        final_move = np.random.randint(len(q_value_np))
    return final_move
  
def Simple_DQN(env, lr = 0.001,episodes=100, max_step = 100,gamma=0.9,test_policy_freq=100):
    '''
      简单DQN算法实现
      核心思想：使用神经网络逼近Q函数，解决表格方法的局限性
    '''
    nS, nA = env.observation_space.n, env.action_space.n
    epsilons = decay_schedule(1,0.01,0.8, episodes)

    # 神经网络模型：简单的线性层，输入状态，输出各动作的Q值
    model = Linear(nS, nA)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    
    for i in range(episodes): 
        state, _ = env.reset()
        state = conv2tensor(state, nS)
        finished = False
        step = 0
        while not finished :
            q_value = model(state)

            # take action
            action = select_action(q_value, epsilons[i])
            next_state, reward, finished, _, _ = env.step(action)
            next_state = conv2tensor(next_state, nS)

            # find target
            target = q_value.clone().detach()
            q_value_next = model(next_state).detach().numpy().squeeze()
            td_target = reward + gamma * q_value_next.max() * (not finished)
            target[action] = torch.tensor(td_target, dtype=torch.float32)            
            optimizer.zero_grad()
            td_error = loss_fn(q_value, target)  # 简易设计，并不合理，合理方式见dqn
            td_error.backward()
            optimizer.step()
            state = next_state

            step += 1
            if step >= max_step:
                break

        if finished:
            results.append(reward)

        
        if (i>0) and (i % test_policy_freq == 0):
            results_array = np.array(results)
            print("Running episode  {} Reaches goal {:.2f}%. ".format(
                i, 
                results_array[-100:].mean()*100))

    return 
  
if __name__ == '__main__':
  env = gym.make('FrozenLake-v1')
  # 19000 Reaches goal 41.00%., max 62
  Simple_DQN(env,lr = 0.001,episodes=20000, max_step = 100,gamma=0.9,test_policy_freq=1000)    

  env = gym.make('FrozenLake-v1',map_name="8x8")
  # 5%
  Simple_DQN(env,lr = 0.001,episodes=20000, max_step = 100,gamma=0.9,test_policy_freq=1000)

  '''
  面临的问题
    Non-stationary target
    No independent and identically distributed
  '''