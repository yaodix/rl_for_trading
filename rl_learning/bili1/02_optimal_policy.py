'''

'''
import random
import numpy as np

class FrozenLake:
    def __init__(self):
      self.reset()
      self.set_tran()
      
    # 设置初始位置和地图
    def reset(self):
        self.position = 0
        self.set_map()
        return self.position
    
    def set_map(self):
        self.map = list(range(16))
        self.map[self.position] = '*'
        
    # 0: left, 1: down, 2: right, 3: up
    def set_tran(self):
            self.transition = {# transition matrix
                0: {  # key: state(position)
                    0: [(0.6666666666666666, 0, 0.0, False), # key:action,
                        (0.3333333333333333, 4, 0.0, False)  # val: (prob, next_state, reward, done)
                    ],
                    1: [(0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 0, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 1, 0.0, False),
                        (0.6666666666666666, 0, 0.0, False)
                    ]
                },
                1: {
                    0: [(0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True)
                    ],
                    1: [(0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 0, 0.0, False)
                    ]
                },
                2: {
                    0: [(0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 6, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 6, 0.0, False),
                        (0.3333333333333333, 3, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 6, 0.0, False),
                        (0.3333333333333333, 3, 0.0, False),
                        (0.3333333333333333, 2, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 3, 0.0, False),
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False)
                    ]
                },
                3: {
                    0: [(0.3333333333333333, 3, 0.0, False),
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 7, 0.0, True)
                    ],
                    1: [(0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 7, 0.0, True),
                        (0.3333333333333333, 3, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 7, 0.0, True),
                        (0.6666666666666666, 3, 0.0, False)
                    ],
                    3: [(0.6666666666666666, 3, 0.0, False),
                        (0.3333333333333333, 2, 0.0, False)
                    ]
                },
                4: {
                    0: [(0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 8, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True)
                    ],
                    2: [(0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 0, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 4, 0.0, False)
                    ]
                },
                5: {
                    0: [(1.0, 5, 0, True)],
                    1: [(1.0, 5, 0, True)],
                    2: [(1.0, 5, 0, True)],
                    3: [(1.0, 5, 0, True)]
                },
                6: {
                    0: [(0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 10, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 7, 0.0, True)
                    ],
                    2: [(0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 7, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 7, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True)
                    ]
                },
                7: {
                    0: [(1.0, 7, 0, True)],
                    1: [(1.0, 7, 0, True)],
                    2: [(1.0, 7, 0, True)],
                    3: [(1.0, 7, 0, True)]
                },
                8: {
                    0: [(0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 12, 0.0, True)
                    ],
                    1: [(0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 12, 0.0, True),
                        (0.3333333333333333, 9, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 12, 0.0, True),
                        (0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 4, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 4, 0.0, False),
                        (0.3333333333333333, 8, 0.0, False)
                    ]
                },
                9: {
                    0: [(0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 13, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 8, 0.0, False),
                        (0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 10, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True)
                    ],
                    3: [(0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 8, 0.0, False)
                    ]
                },
                10: {
                    0: [(0.3333333333333333, 6, 0.0, False),
                        (0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 11, 0.0, True)
                    ],
                    2: [(0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 11, 0.0, True),
                        (0.3333333333333333, 6, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 11, 0.0, True),
                        (0.3333333333333333, 6, 0.0, False),
                        (0.3333333333333333, 9, 0.0, False)
                    ]
                },
                11: {
                    0: [(1.0, 11, 0, True)],
                    1: [(1.0, 11, 0, True)],
                    2: [(1.0, 11, 0, True)],
                    3: [(1.0, 11, 0, True)]
                },
                12: {
                    0: [(1.0, 12, 0, True)],
                    1: [(1.0, 12, 0, True)],
                    2: [(1.0, 12, 0, True)],
                    3: [(1.0, 12, 0, True)]
                },
                13: {
                    0: [(0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 12, 0.0, True),
                        (0.3333333333333333, 13, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 12, 0.0, True),
                        (0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False)
                    ],
                    2: [(0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 9, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 9, 0.0, False),
                        (0.3333333333333333, 12, 0.0, True)
                    ]
                },
                14: {
                    0: [(0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False)
                    ],
                    1: [(0.3333333333333333, 13, 0.0, False),
                        (0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 15, 1.0, True)
                    ],
                    2: [(0.3333333333333333, 14, 0.0, False),
                        (0.3333333333333333, 15, 1.0, True),
                        (0.3333333333333333, 10, 0.0, False)
                    ],
                    3: [(0.3333333333333333, 15, 1.0, True),
                        (0.3333333333333333, 10, 0.0, False),
                        (0.3333333333333333, 13, 0.0, False)
                    ]
                },
                15: {
                    0: [(1.0, 15, 0, True)],
                    1: [(1.0, 15, 0, True)],
                    2: [(1.0, 15, 0, True)],
                    3: [(1.0, 15, 0, True)]
                }
            }

    def show(self):
        print(f"state: {self.position}")
        self.set_map()
        for i, s in enumerate(self.map):
            print("| ", end="")
            if s == "*":
                print(s, "".rjust(4), end=" ")
            else:
                print(str(s).zfill(2), "".rjust(3), end=" ")
            if (i + 1) % 4 == 0: print("|")

    def step(self, action):
        '''
            action: 0: left, 1: down, 2: right, 3: up
        '''
        node = self.transition[self.position][action]
        probs,states,rewards,dones = zip(*node)
        choice = random.choices(population=states,weights=probs,k=1)[0]
        i = states.index(choice)
        self.position = states[i]
        return states[i], rewards[i], dones[i]       
    
    
class RandomAgent:
    def __init__(self):
        #self.policy_dict = {k:v for k in range(16) for v in random.choices(population=range(4),k=16)}#  wrong
        # 为每个状态分配一个随机动作 (0-3)
        self.policy_dict = {k: random.choice(range(4)) for k in range(16)}

    def act(self, state):
        '''
        act 的 Docstring
        :param self: RandomAgent 实例
        :param state: 当前状态
        :return: 随机选择的动作
        '''
        action = self.policy_dict[state]
        return action
    
def test_game(env, pi, n_episodes=100, max_steps=100):
    results = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = pi(state)
            state, reward, done = env.step(action)
            steps += 1
        results.append(reward > 0)
    return np.sum(results) / len(results)
    
def print_policy(pi, n_cols=4):
    print('Policy:')
    arrs = {k:v for k,v in enumerate(('<', 'v', '>', '^'))}
    nS = 16
    for s in range(nS):
        a = pi(s)
        print("| ", end="")
        if s in [5,7,11,12,15]:
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

class Human_Agent:
    def __init__(self):
        LEFT, DOWN, RIGHT, UP = range(4)
        self.policy_dict = {
        0:RIGHT, 1:RIGHT, 2:DOWN, 3:LEFT,
        4:DOWN, 5:LEFT, 6:DOWN, 7:LEFT,
        8:RIGHT, 9:RIGHT, 10:DOWN, 11:LEFT,
        12:LEFT, 13:RIGHT, 14:RIGHT, 15:LEFT
        }
    
    def action(self, state):
        return self.policy_dict[state]
    
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    '''
    迭代方式计算策略评估，直到价值函数收敛。策略已经固定在pi中
    使用迭代方式进行评估是因为这是求解贝尔曼期望方程的标准方法。
    '''
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]: #只计算当前策略选择的动作
                # 这里prob已知，相当于policy已知。但是很多时候agent不知道policy，只有与环境
                # 交互的能力, 下面方程对应 状态价值函数的贝尔曼期望方程
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done)) 
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V


def Q_function(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])): # 评估所有可能动作的价值，以便选择最优动作
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    return Q


def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])): #计算所有动作的价值
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))  # 从V计算Q
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

# 这里为什么policy_evaluation用Value值评估，而不是Q值评估？
# 因为Q值评估需要知道当前状态和动作，而policy_evaluation只需要知道当前状态。
# 所以用Value值评估更简单。
# 为什么用Q值更新策略？
# 因为Q值评估是对所有动作的期望，而policy_evaluation只需要知道当前状态的Value值。
# 所以用Q值更新策略更简单。
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(4, 16)
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
    optimize_steps = 0
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        optimize_steps += 1
        if old_pi == {s:pi(s) for s in range(len(P))}:
            break
        print(f"optimize_steps: {optimize_steps}")
    return V, pi



if __name__ == "__main__":
   
    env = FrozenLake()
    
    # agent = RandomAgent()
    # print_policy(agent.act)
    h_agent= Human_Agent()
    print_policy(h_agent.action)

    res = test_game(env, h_agent.action)
    print(res)

    value = policy_evaluation(h_agent.action, env.transition)
    print(value.reshape(4,4))

    Q = Q_function(value, env.transition)
    print(Q)
    
    new_policy = policy_improvement(value, env.transition)

    Value, Pi = policy_iteration(env.transition)
    print(Value.reshape(4,4))
    print_policy(Pi)
    test_res = test_game(env, Pi)  # 0.75
    print(test_res)
