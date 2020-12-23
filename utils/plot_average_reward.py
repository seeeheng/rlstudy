class RewardPlotter(self):
	def __init__(self):
		total_reward = 0
		average_reward_plot = []

	def remember(self, episode_reward, i_episode):
		self.total_reward += episode_reward
		self.average_reward_plot += [[i_episode, total_reward/(i_episode+1)],]

	def plot(self):
		# Plotting and evaluation
		xs = [x[0] for x in self.average_reward_plot]
		ys = [y[1] for y in self.average_reward_plot]
		# plt.ylim((0,average_reward+average_reward*0.2))

		fig, ax = plt.subplots()
		ax.plot(xs,ys)
		ax.set_ylabel('Average Reward So far')
		ax.set_xlabel('Number of Episodes')
		ax.set_title('MC-glie for Blackjack')
		plt.show()