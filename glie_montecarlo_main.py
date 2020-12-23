import numpy as np
import gym
import matplotlib.pyplot as plt
from utils.Buffer import Buffer
from montecarlo.monte_carlo_agent import MonteCarloGlie
from utils.plot_policy import plot_policy
from utils.plot_average_reward import RewardPlotter
from utils.GymEnvironment import GymEnvironment

GAMMA = 0
N_EPISODES = 500000
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.02

def main():
	env = GymEnvironment("Blackjack-v0")
	print("Number of Actions = {}.\n0 = Stick, 1 = Hit\n".format(env.env.action_space.n))
	print("Shape of states = {}".format(len(env.env.observation_space)))
	current_state = [int(x) for x in env.reset()]
	print("Sample Observation:\n(player's current sum, dealer's one showing card, usable ace)\n{}\n=======================".format(current_state))

	mc = MonteCarloGlie(Buffer())
	epsilon = 1
	reward_plotter = RewardPlotter()
	# Begin learning!
	for i_episode in range(N_EPISODES):
		epsilon *= EPSILON_DECAY
		done = False
		while not done: # for episode
			current_state, done = mc.step(current_state, epsilon)
		# end of episode
		episode_reward = mc.learn()
		reward_plotter.remember(episode_reward, i_episode)
		mc.reset_memory()
		current_state = [int(x) for x in env.reset()] # RESET THE ENV, and reset current state.

	reward_plotter.plot()
	# last_stick = {11:None,12:None,13:None,14:None, 15:None, 16:None, 17:None, 18:None, 19:None, 20:None, 21:None}

	# STATE: player's current sum, dealer's one showing card, usable ace
	# 32, 11, 2, 2

if __name__ == "__main__":
	main()