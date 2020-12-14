import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

EPSILON = 1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.3
GAMMA = 0.9
N_EPISODES = 30000
PRINT_EVERY = 100

"""
Actions
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""
# stepping would return a tuple
# (state, reward, done, _)
# print(env.render())

class QLearning:
	def __init__(self, n_states, n_actions, env, epsilon=1, lr=0.02, gamma=0.9):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.lr = lr
		self.gamma = gamma
		self.env = env
		
		self.episode_reward = 0
		self.total_reward = 0
		self.q = np.full([n_states, n_actions],5.0) # initializing with positive value to encourage exploration
		self.current_state = None

	def act(self, current_state):
		current_argmax = np.argmax(self.q[current_state][:])
		# Initializing all actions with probabilities
		# Greedy action will be tweaked later. Rest of the actions = e/|nA|
		probs = np.ones(self.n_actions) * (self.epsilon/self.n_actions) # epsilon / |number of actions|
		probs[current_argmax] += 1-self.epsilon
		action = np.random.choice(np.arange(self.n_actions), p=probs)
		return action

	def first_step(self, current_state):
		self.episode_reward = 0
		self.current_state = current_state

	def learn(self, current_state, next_state, current_action, reward):
		self.q[current_state][current_action] += self.lr * (reward + self.gamma * np.max(self.q[next_state]) - self.q[current_state][current_action])

	def step(self):
		current_action = self.act(self.current_state)
		next_state, reward, done, _ = self.env.step(current_action)
		self.learn(self.current_state, next_state, current_action, reward)
		self.current_state = next_state
		self.episode_reward += reward
		# print("Current reward={}".format(self.episode_reward))
		# env.render()
		if done:
			self.total_reward += self.episode_reward
			self.epsilon = max(self.epsilon*EPSILON_DECAY,0.05)
		return done

if __name__ == "__main__":
	env = gym.make('Taxi-v3')
	print("Number of States = {}".format(env.nS))
	print("Number of Actions = {}".format(env.nA))
	current_state = env.reset()

	q_agent = QLearning(env.nS, env.nA, env, epsilon=EPSILON, lr=LEARNING_RATE, gamma=GAMMA)

	scores_window = deque(maxlen=PRINT_EVERY)
	average_reward_plot = []
	best_average_reward = -np.inf
	for i_episode in range(N_EPISODES):
		q_agent.first_step(current_state)
		done = False
		while not done:
			done = q_agent.step()
			# print("--------------------------------------------")
		current_state = env.reset()

		# Printing stuff		
		scores_window.append(q_agent.episode_reward)
		running_average = np.mean(scores_window)
		if i_episode % PRINT_EVERY == 0:
			average_reward_plot += [[i_episode, running_average],]
			if running_average > best_average_reward:
				best_average_reward = running_average
			print("Episode {}/{}, best_average_reward={:.2f}, total_average={:.2f}\r".format(i_episode+1,N_EPISODES,best_average_reward,q_agent.total_reward/(i_episode+1)), end="")

		if running_average >= 9.7:
			best_average_reward = running_average
			print("SOLVED in {} episodes!".format(i_episode))
			break


	print()
	print("End of training, saving agent...")
	np.save("q-agent",q_agent.q)

	# Plotting and evaluation
	xs = [x[0] for x in average_reward_plot]
	ys = [y[1] for y in average_reward_plot]
	# plt.ylim((0,average_reward+average_reward*0.2))

	fig, ax = plt.subplots()
	ax.plot(xs,ys)
	ax.set_ylabel('Average Reward So far')
	ax.set_xlabel('Number of Episodes')
	ax.set_title('Sarsa for Taxi')
	plt.show()
