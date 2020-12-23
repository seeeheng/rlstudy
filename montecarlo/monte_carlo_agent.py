class MonteCarloGlie:
	def __init__(self, env, buffer):
		self.env = env
		self.n = np.zeros([32,11,2,2]) # starting up according to state
		self.q = np.zeros([32,11,2,2])
		self.memory = buffer
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
		obs, reward, done, _ = self.env.step(action)
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