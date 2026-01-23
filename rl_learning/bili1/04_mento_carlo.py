'''
蒙特卡罗方法的本质：
  通过完整的回合轨迹来估计价值函数
  基于大数定律：当样本足够多时，样本均值收敛于期望值
  无需环境模型：直接从经验中学习

蒙特卡罗方法的缺点：
  要等到游戏一轮完结后才更新
  利用的信息中噪声较多，学习效率较低

'''

from help import FrozenLake, print_policy, test_game
import numpy as np

def select_action(state, Q, mode="both"):
    if mode == "explore":
        return np.random.randint(len(Q[state]))
    if mode == "exploit":
        return np.argmax(Q[state])
    if mode == "both":
        if np.random.random() > 0.5:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))
          
          
def play_game(env, Q ,max_steps=200):
    state = env.reset()
    episode = []
    finished = False
    step = 0

    while not finished:
        action = select_action(state, Q, mode='both')
        next_state, reward, finished = env.step(action)
        experience = (state, action, finished, reward)
        episode.append(experience)
        if step >= max_steps:
            break
        state = next_state
        step += 1

    return np.array(episode,dtype=object)
  
def monte_carlo(env, episodes=10000, test_policy_freq=1000):
    nS, nA = 16, 4   # FrozenLake环境有16个状态，4个动作
    Q = np.zeros((nS, nA), dtype=np.float64)  # 这就是Q表
    returns = {} 

    for i in range(episodes): 
        episode = play_game(env, Q)
        visited = np.zeros((nS, nA), dtype=bool)

        for t, (state, action, _, _) in enumerate(episode):
            state_action = (state, action)
            if not visited[state][action]:
                visited[state][action] = True
                discount = np.array([0.9**i for i in range(len(episode[t:]))])
                reward = episode[t:, -1]
                G = np.sum(discount * reward)
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]  

                Q[state][action] = sum(returns[state_action]) / len(returns[state_action])
                #Q[state][action] = Q[state][action] + 1/len(returns[state_action]) * (G - Q[state][action])
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

        if i % test_policy_freq == 0:
                print("Test episode {} Reaches goal {:.2f}%. ".format
                (i, test_game(env, pi)*100))
            
    return pi, Q
  
  
if __name__ == "__main__":
    env = FrozenLake()
    policy_mc, Q = monte_carlo(env, episodes=20000) # Test episode 19000 Reaches goal 63.00%.max 91%
    print_policy(policy_mc)
    print('Reaches goal {:.2f}%. '.format(test_game(env, policy_mc)*100))
    