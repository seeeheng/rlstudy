import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

EPSILON = 1
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.5
GAMMA = 0.9
N_EPISODES = 200
PRINT_EVERY = 1
LAMBDA = 0.8

env = gym.make('Taxi-v3')
print("Number of States = {}".format(env.nS))
print("Number of Actions = {}".format(env.nA))
current_state = env.reset()

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

class sarsalambda:
	def __init__(self, n_states, n_actions, env, epsilon=1, lr=0.02, gamma=0.9, param_lambda=1):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.lr = lr
		self.gamma = gamma
		self.env = env
		self.param_lambda = param_lambda

		self.episode_reward = 0
		self.total_reward = 0
		self.q = np.full([n_states, n_actions],5.0)
		self.e = np.zeros([n_states, n_actions])
		self.current_action = None
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
		self.current_action = self.act(current_state)

	def learn(self, current_state, next_state, current_action, next_action, reward):
		q_sa = self.q[current_state][current_action]
		q_sa_next = self.q[next_state][next_action]
		delta = reward + (self.gamma * q_sa_next) - q_sa
		self.e[current_state][current_action] += 1

		for one_state in range(self.n_states):
			for one_action in range(self.n_actions):
				self.q[one_state][one_action] += self.lr * delta * self.e[one_state][one_action]
				self.e[one_state][one_action] *= self.param_lambda * self.gamma

	def step(self):
		next_state, reward, done, _ = self.env.step(self.current_action)
		next_action = self.act(next_state) # getting next action from policy.
		self.learn(self.current_state, next_state, self.current_action, next_action, reward)
		self.current_action = next_action
		self.current_state = next_state
		self.episode_reward += reward
		# print("Current reward={}".format(self.episode_reward))
		# env.render()
		
		if done:
			self.total_reward += self.episode_reward
			self.epsilon = max(self.epsilon*EPSILON_DECAY,0.05)
			# print("{}\r".format(self.epsilon), end="")
		return done

if __name__ == "__main__":
	td_agent = sarsalambda(env.nS, env.nA, env, epsilon=EPSILON, lr=LEARNING_RATE, gamma=GAMMA, param_lambda=LAMBDA)

	scores_window = deque(maxlen=PRINT_EVERY)
	average_reward_plot = []
	best_average_reward = -99999999999
	for i_episode in range(N_EPISODES):
		td_agent.first_step(current_state)
		done = False
		while not done:
			done = td_agent.step()
			# print("--------------------------------------------")
		scores_window.append(td_agent.episode_reward)
		running_average = np.mean(scores_window)
		if i_episode % PRINT_EVERY == 0:
			average_reward_plot += [[i_episode, running_average],]
			if running_average > best_average_reward:
				best_average_reward = running_average
			print("Episode {}/{}, best_average_reward={:.2f}, total_average={:.2f}\r".format(i_episode+1,N_EPISODES,best_average_reward,td_agent.total_reward/(i_episode+1)), end="")

		current_state = env.reset()

	print()
	print("End of training, saving agent...")
	np.save("sarsalambda-agent",td_agent.q)

	# Plotting and evaluation
	xs = [x[0] for x in average_reward_plot]
	ys = [y[1] for y in average_reward_plot]
	# plt.ylim((0,average_reward+average_reward*0.2))

	fig, ax = plt.subplots()
	ax.plot(xs,ys)
	ax.set_ylabel('Average Reward So far')
	ax.set_xlabel('Number of Episodes')
	ax.set_title('sarsalambda for Taxi')
	plt.show()
