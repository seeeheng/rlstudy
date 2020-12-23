# dynamic programming
import numpy as np
import gym
import matplotlib.pyplot as plt

environment_name ='FrozenLake-v0' 
env = gym.make(environment_name)
state = env.reset()
env.render()

# Probabilities of transition can be gotten.
print("Number of states in environment {}: {}".format(environment_name, env.nS))
print("Number of actions in environment {}: {}".format(environment_name, env.nA))

# for i in range(10):
#     # print(env.P[])
#     s,r,d,prob = env.step(np.random.randint(4))
#     env.render()
#     print("State: {}".format(s))
#     print("Reward: {}".format(r))
#     print("Done: {}".format(d))
#     print("Prob = {}".format(prob))

#     if d:
#         env.reset()


class ValueFunction:
    def __init__(self, env, gamma=0.9, theta=0.01):
        self.env = env
        self.n_states = env.nS
        self.value_function = np.zeros(self.n_states)
        self.gamma = gamma
        self.theta = theta
    def show_value_function(self):
        print(self.value_function)
        return self.value_function

    def act(self, current_state):
        possible_actions = [0,1,2,3]
        action_dict = {}
        for action in possible_actions:
            pi_state = 0
            possibilities = self.env.P[current_state][action]
            for poss in possibilities:
                prob = poss[0]
                next_state = poss[1]
                reward = poss[2]
                done = poss[3]
                pi_state += prob * (reward + self.gamma * self.value_function[next_state])
            action_dict[action] = pi_state
        best_action = max(action_dict, key=action_dict.get) 
        return best_action

    def evaluate_best_value(self,current_state,best_action):
        possible_actions = [0,1,2,3]
        action_dict = {}
        possibilities = self.env.P[current_state][best_action]
        pi_state = 0
        for poss in possibilities:
            prob = poss[0]
            next_state = poss[1]
            reward = poss[2]
            done = poss[3]
            pi_state += prob * (reward + self.gamma * self.value_function[next_state])
        return pi_state

    def evaluate_value(self):
        remaining_states = np.array(range(self.n_states))
        i = 0
        while len(remaining_states)>0:
            i+=1
            # print("Iteration {}".format(i))
            delta = 0
            for state in range(16):
                # print("Evaluating state {}".format(state))
                current_value = self.value_function[state]
                best_action = self.act(state)
                new_value = self.evaluate_best_value(state,best_action)
                self.value_function[state] = new_value
                delta = max(delta, abs(current_value - new_value))
                # self.show_value_function()
                if delta < self.theta:
                    remaining_states = remaining_states[remaining_states != state]


n_episodes = 10000
cumulative_reward = 0
reward_plot = []
for i in range(n_episodes):
    env_terminate = False
    vf = ValueFunction(env)
    vf.evaluate_value()
    episode_reward = 0
    while env_terminate != True:
        # env.render()
        best_action = vf.act(state)
        # print("Moving {}".format(['Left','Down','Right','Up'][best_action]))
        state,reward,env_terminate, _ = env.step(best_action)
        episode_reward += reward
    cumulative_reward += episode_reward
    average_reward = cumulative_reward/(i+1)
    reward_plot += [(i,average_reward)]
    # env.render()
    env.reset()
average_reward = cumulative_reward/n_episodes
print("Average Reward earned: {}".format(average_reward))
xs = [x[0] for x in reward_plot]
ys = [y[1] for y in reward_plot]
plt.plot(xs, ys)
plt.ylabel('Average Reward So Far')
plt.xlabel('Number of Episodes')
plt.title('Dynamic Programming (VALUE ITERATION) for Frozen Lake')
plt.ylim((0,average_reward+average_reward*0.2))
plt.show()
# 33% chance of going to 