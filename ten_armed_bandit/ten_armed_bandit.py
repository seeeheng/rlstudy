class TenArmedBandit:
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