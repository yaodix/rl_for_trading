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
def select_action(state, Q, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(len(Q[state]))
      
      
def dyna_q(env,episodes=100,gamma=0.9,n_planning = 3,test_policy_freq=1000):
    nS, nA = 16,4
    Q = np.zeros((nS, nA), dtype=np.float64)
    alphas = decay_schedule(0.5,0.01,0.5, episodes)
    epsilons = decay_schedule(1,0.01,0.8, episodes)

    T_count = np.zeros((nS, nA, nS), dtype=np.int32)
    R_model = np.zeros((nS, nA, nS), dtype=np.float64)
    planning_track = []

    for i in range(episodes): 
        state = env.reset()
        finished = False
        while not finished:
            action = select_action(state, Q, epsilons[i])
            next_state, reward, finished = env.step(action)

            #  记录环境反馈信息，统计转移次数和回报
            T_count[state][action][next_state] += 1
            r_diff = reward - R_model[state][action][next_state]
            R_model[state][action][next_state] += (r_diff / T_count[state][action][next_state])

            target = reward + gamma * Q[next_state].max() * (not finished)
            error = target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[i] * error

            backup_next_state = next_state
            # 进入规划循环
            for _ in range(n_planning):
                if Q.sum() == 0: break
                # 选择一个曾经进入过的状态
                visited_states = np.where(np.sum(T_count, axis=(1, 2)) > 0)[0]
                state = np.random.choice(visited_states)
                # 选择一个曾经选择过的行动
                actions_taken = np.where(np.sum(T_count[state], axis=1) > 0)[0]
                action = np.random.choice(actions_taken)
                # 根据环境模型计算出可能的下一步状态和可能的回报
                probs = T_count[state][action]/T_count[state][action].sum()
                next_state = np.random.choice(np.arange(nS), size=1, p=probs)[0]
                reward = R_model[state][action][next_state]
                planning_track.append((state, action, reward, next_state))

                target = reward + gamma * Q[next_state].max()
                error = target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[i] * error
            
            state = backup_next_state


        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
        if i % test_policy_freq == 0:
                print("Test episode {} Reaches goal {:.2f}%. ".format
                (i, test_game(env, pi)*100))

    return pi,Q
  
def trajectory_sampling(env,episodes=100,gamma=0.9,
                        planning_freq = 5,
                        max_trajectory_depth=100,
                        test_policy_freq=1000):
    nS, nA = 16,4
    Q = np.zeros((nS, nA), dtype=np.float64)
    alphas = decay_schedule(0.5,0.01,0.5, episodes)
    epsilons = decay_schedule(1,0.01,0.8, episodes)

    T_count = np.zeros((nS, nA, nS), dtype=np.int32)
    R_model = np.zeros((nS, nA, nS), dtype=np.float64)
    planning_track = []

    for i in range(episodes): 
        state = env.reset()
        finished = False
        while not finished:
            action = select_action(state, Q, epsilons[i])
            next_state, reward, finished = env.step(action)

            #  记录环境反馈信息，统计转移次数和回报
            T_count[state][action][next_state] += 1
            r_diff = reward - R_model[state][action][next_state]
            R_model[state][action][next_state] += (r_diff / T_count[state][action][next_state])

            target = reward + gamma * Q[next_state].max() * (not finished)
            error = target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[i] * error

            backup_next_state = next_state
            # 进入规划循环
            if i % planning_freq == 0:
                for _ in range(max_trajectory_depth):
                    if Q.sum() == 0: break
                    # 从当前实际的状态进行规划
                    action = Q[state].argmax()
                    if not T_count[state][action].sum(): break
                    probs = T_count[state][action]/T_count[state][action].sum()
                    next_state = np.random.choice(np.arange(nS), size=1, p=probs)[0]
                    reward = R_model[state][action][next_state]
                    planning_track.append((state, action, reward, next_state))

                    target = reward + gamma * Q[next_state].max()
                    error = target - Q[state][action]
                    Q[state][action] = Q[state][action] + alphas[i] * error

                    state = next_state
            
            state = backup_next_state


        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
        if i % test_policy_freq == 0:
                print("Test episode {} Reaches goal {:.2f}%. ".format
                (i, test_game(env, pi)*100))

    return pi,Q
  
  
if __name__ == '__main__':
  env = FrozenLake()
  policy_qlearning, Q_qlearning = dyna_q(env,episodes=20000)
  
  
  policy_qlearning, Q_qlearning = trajectory_sampling(env,episodes=20000)