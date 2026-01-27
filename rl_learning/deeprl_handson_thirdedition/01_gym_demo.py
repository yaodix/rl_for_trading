

import gymnasium as gym

e = gym.make('CartPole-v1')
obs, info = e.reset()

print(obs)
print(info)

print(f"e.action_space {e.action_space}")
print(f"e.obs_space {e.observation_space}")