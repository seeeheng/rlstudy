class TenArmedAgent:
	def __init__(self, ten_armed_bandit):
		self.q_table = np.zeros(10)
		self.n_table = np.zeros(10)
		self.tb = ten_armed_bandit

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