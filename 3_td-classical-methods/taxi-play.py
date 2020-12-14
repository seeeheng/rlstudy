import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
from sarsa import Sarsa

N_EPISODES=20

env = gym.make('Taxi-v3')
print("Number of States = {}".format(env.nS))
print("Number of Actions = {}".format(env.nA))
current_state = env.reset()

q = np.load("q-agent.npy")
td_agent = Sarsa(env.nS, env.nA, env)
td_agent.q = q

scores_window = deque(maxlen=10)
for i_episode in range(N_EPISODES):
	current_state = env.reset()	
	done = False
	episode_reward = 0
	while not done:
		next_state, reward, done, _ = env.step(np.argmax(td_agent.q[current_state][:]))
		episode_reward += reward
		current_state = next_state
		env.render()
		print()
	episode_reward += reward
	scores_window.append(episode_reward)
	running_average = sum(scores_window) / len(scores_window)
	print("Episode {}/{}, running_average={:.2f}, total_average={:.2f}\r".format(i_episode+1,N_EPISODES,running_average,td_agent.total_reward/(i_episode+1)), end="")
	print()
	print("-----")