import gymnasium as gym
import time

env = gym.make("LunarLander-v3", render_mode="human")
state, _ = env.reset()
env.render()
time.sleep(1)
env.close()
