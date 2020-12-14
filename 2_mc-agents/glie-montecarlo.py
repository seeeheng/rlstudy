import numpy as np
import gym
import matplotlib.pyplot as plt

GAMMA = 0
N_EPISODES = 500000
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.02

env = gym.make("Blackjack-v0")
print("Number of Actions = {}.\n0 = Stick, 1 = Hit\n".format(env.action_space.n))
print("Shape of states = {}".format(len(env.observation_space)))
current_state = [int(x) for x in env.reset()]
print("Sample Observation:\n(player's current sum, dealer's one showing card, usable ace)\n{}\n=======================".format(current_state))

class Buffer:
	def __init__(self):
		# stores in state(t), action(t), reward(t+1)
		self.memory = [] 

	def add_into_memory(self, stuff):
		self.memory += [stuff]

	def get_and_clear_memory(self):
		# This function gets called at the end of each episode, when the monte-carlo agent is learning from the episodes.
		memory = self.memory
		self.memory = []
		return memory

class MonteCarloGlie:
	def __init__(self, env):
		self.env = env
		self.n = np.zeros([32,11,2,2]) # starting up according to state
		self.q = np.zeros([32,11,2,2])
		self.memory = Buffer()
		self.reward = 0
		self.random = 0
		self.lr = LEARNING_RATE

	def reset_memory(self):
		self.reward = 0

	def act(self, state, epsilon):
		current_argmax = np.argmax(self.q[tuple(state)])
		probs = np.ones(2) * (epsilon/2) # epsilon / |number of actions|
		probs[current_argmax] = 1-epsilon + (epsilon/2)
		print(probs)
		action = np.random.choice([0,1], p=probs)
		## THIS IMPLEMENTATION IS NOT CORRECT. ##
		# if np.random.rand() > epsilon:
		# 	action = np.argmax(self.policy[tuple(state)])
		# else:
		# 	print("I'm taking a chance...")
		# 	self.random += 1
		# 	action = np.random.choice([0,1])
		print("I'm going to {}".format(['Stick!','Hit!'][action]))
		return action

	def step(self,current_state, epsilon):
		action = self.act(current_state, epsilon)
		obs, reward, done, _ = env.step(action)
		obs = [int(x) for x in obs]
		self.memory.add_into_memory([current_state,action,reward])
		return obs, done

	def _get_reward(self):
		return self.reward
	def _reset_reward(self):
		self.reward = 0
	def _add_to_reward(self, reward):
		self.reward += reward

	def learn(self):
		memory_buffer = self.memory.get_and_clear_memory() # gets memory from the buffer then clears everything
		print("memory_buffer = {}".format(memory_buffer))
		t = 0 # maintaining a counter in order to properly deal with discounts.
		returns = 0
		for i in range(len(memory_buffer)-1,-1,-1): # maintain a decreasing index to go backwards
			memory = memory_buffer[i]
			print("Current memory = {}".format(memory))
			state = [x-1 for x in memory[0]] # have to -1 since the dealer shows cards (1-10, 1 = ace) and player shows a sum(1-32)
			returns *= (GAMMA **(t))
			reward = memory[2]
			returns += memory[2] # adding current reward
			action = memory[1]
			sa = tuple(state + [action]) # putting state + action together into one tuple, for lookup
			self.n[sa] += 1
			print("self.n[sa] = {}".format(self.n[sa]))
			print("self.q[sa] = {}".format(self.q[sa]))
			# Monte Carlo Update!
			update = self.lr*(1/self.n[sa])*(returns-self.q[sa])
			print("Update = {}".format(update))
			self.q[sa] += update
			# Updating reward for tracking performance
			if reward: 
				self._add_to_reward(reward)
			t += 1
		return self._get_reward()

mc = MonteCarloGlie(env)
total_reward = 0
average_reward_plot = []
epsilon = 1

# Begin learning!
for i_episode in range(N_EPISODES):
	epsilon *= EPSILON_DECAY
	# epsilon = 1/(i_episode+1)
	done = False
	while not done: # for episode
		current_state, done = mc.step(current_state, epsilon)
	# end of episode
	episode_reward = mc.learn()
	total_reward += episode_reward
	average_reward_plot += [[i_episode, total_reward/(i_episode+1)],]
	mc.reset_memory()
	current_state = [int(x) for x in env.reset()] # RESET THE ENV, and reset current state.
	print()

# exit()

# Plotting and evaluation
xs = [x[0] for x in average_reward_plot]
ys = [y[1] for y in average_reward_plot]
# plt.ylim((0,average_reward+average_reward*0.2))

fig, ax = plt.subplots()
ax.plot(xs,ys)
ax.set_ylabel('Average Reward So far')
ax.set_xlabel('Number of Episodes')
ax.set_title('MC-glie for Blackjack')

# last_stick = {11:None,12:None,13:None,14:None, 15:None, 16:None, 17:None, 18:None, 19:None, 20:None, 21:None}
def plot_policy(q):
	"""
	Recall that in order to use numpy arrays to store the information, the indexes are one less than the sum.
	i.e., q[0][10] corresponds to a player sum of 1, a dealer first card of 11
	"""
	print("NO USABLE ACE")
	for i in range(10,21):
		q_player = q[i] 
		for j in range(10):
			action = np.argmax(q_player[j][0])
			if action == 0:
				# last_stick[i] = j
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Stick"))
			else:
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Hit"))
		print()

	print("USABLE ACE")
	for i in range(10,21):
		q_player = q[i] 
		for j in range(10):
			action = np.argmax(q_player[j][1])
			if action == 0:
				# last_stick[i] = j
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Stick"))
			else:
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Hit"))
		print()
plot_policy(mc.q)
np.save("./q",mc.q)
print(mc.random)
plt.show()

# STATE: player's current sum, dealer's one showing card, usable ace
# 32, 11, 2, 2