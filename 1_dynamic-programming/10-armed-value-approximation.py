import numpy as np

class ten_armed_bandit:
	def __init__(self, seed=0):
		self.seed = seed
		np.random.seed(seed=self.seed)
		self.arms = np.array([])

		for i in range(10):
			current = np.random.normal(0,1)
			self.arms = np.append(self.arms, current)

	def show_rewards(self):
		print("Rewards generated = ")
		print(self.arms)

	def pull_arm(self, arm):
		if (arm>9) or (arm<0):
			return 
		mean = self.arms[arm-1]
		reward = np.random.normal(mean,1)
		# print("Arm {} pulled, you get {} reward.".format(arm, reward))
		return reward

class ten_armed_agent:
	def __init__(self):
		self.q_table = np.zeros(10)
		self.n_table = np.zeros(10)
		self.tb = ten_armed_bandit()

	def show_q(self):
		print(self.q_table)
		return self.q_table

	def step(self):
		for n_arm in range(10):
			self.n_table[n_arm] += 1
			n = self.n_table[n_arm]
			q = self.q_table[n_arm]
			r = self.tb.pull_arm(n_arm)
			self.q_table[n_arm] = q + (1/n) * (r - q)

agent = ten_armed_agent()
for i in range(10000):
	agent.step()
	agent.show_q()

# SIMPLE BANDIT ALGORITHM
# Q(A) <- Q(A) 1/ N(A) [ R - Q(A)]



