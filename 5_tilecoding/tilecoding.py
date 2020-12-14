import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt


PRINT_EVERY = 100
N_EPISODES = 2000

# In reference to Udacity's DRL course.

env = gym.make('MountainCar-v0')

# Explore state (observation) space
# print("Environment lows\t", env.observation_space.low)
# print("Environment highs\t", env.observation_space.high)

def create_tiling_grid(low, high, bins):
    n_states = len(low)
    return [np.linspace(low[i],high[i],bins)[1:-1] for i in range(n_states)]

sample_grid = create_tiling_grid(env.observation_space.low,env.observation_space.high,10)
# NOTE: If there are 10 bins, there will only be 9 different indexes. 

sample_state = env.reset()
# print("Sample state: {}".format(sample_state))
# print("Sample grid: {}".format(sample_grid))

def discretize(sample, grid):
    return [np.digitize(sample[i],grid[i]) for i in range(len(sample))]
# print("Sample state discretized: {}".format(discretize(sample_state, sample_grid)))

##########################################################################################################

class TileQAgent:
    def __init__(self, n_actions, env, low, high, bins, epsilon=1, lr=0.02, gamma=0.9):
        self.n_actions = n_actions
        self.env = env
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

        self.episode_reward = 0
        self.total_reward = 0
        self.current_state = None
        self.q = np.zeros([bins-1,bins-1,n_actions])
        self.grid = self.create_tiling_grid(low, high, bins) # FOR NOW, use the same number of bins for all states.

    def create_tiling_grid(self, low, high, bins):
        n_states = len(low)
        return [np.linspace(low[i],high[i],bins)[1:-1] for i in range(n_states)]

    def discretize(self, sample):
        return [np.digitize(sample[i],self.grid[i]) for i in range(len(sample))]

    def act(self, current_state):
        current_argmax = np.argmax(self.q[current_state])
        probs = np.ones(self.n_actions) * (self.epsilon/self.n_actions)
        probs[current_argmax] += (1-self.epsilon)
        action = np.random.choice(np.arange(self.n_actions),p=probs)
        return action

    def learn(self, current_state, next_state, current_action, reward):
        discretized_next_state = self.discretize(next_state)
        current_index = np.append(current_state, current_action)
        print(current_index)
        print(self.q[0])
        # print(self.q[list(current_index)])
        # print(self.q[current_index])
        # print(self.q[0],1,2])
        # print(self.q[current_state][current_action])
        exit()
        self.q[current_state][current_action] += self.lr * (reward + self.gamma * np.max(self.q[discretized_next_state]) - self.q[current_state][current_action]) 

    def first_step(self, current_state):
        self.episode_reward = 0
        self.current_state = current_state

    def step(self):
        discretized_state = self.discretize(self.current_state)
        current_action = self.act(discretized_state)
        next_state, reward, done, _ = self.env.step(current_action)
        self.learn(discretized_state, next_state, current_action, reward)
        self.current_state = next_state
        self.episode_reward += reward
        if done:
            self.total_reward += self.episode_reward
            self.epsilon = max(self.epsilon*EPSILON_DECAY, 0.1)
        return done

current_state = env.reset()
q_agent = TileQAgent(3, env, env.observation_space.low, env.observation_space.high, 10)

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


# def create_tiling_grid(low, high, bins, offsets):
#     # low/high refers to the lowest/highest values that will be required for the tiles
#     # bins refers to the number of bins you'd like to create for the tile
#     # offsets are the separate tiles that are created, above and below your tile.

#     # +1 then [1:-1] is to get rid of low and high dim for buckets
#     n_states = len(bins)
#     grid = [np.linspace(low[dim], high[dim], bins[dim]+1)[1:-1] + offsets[dim] for dim in range(n_states)]
#     return grid


# def create_tilings(low, high, bins, offsets):
# # this function allows for creation of multiple tiles at the same time, according to the bins and offsets.
#     return [create_tiling_grid(low, high, bins[n], offsets[n]) for n in range(len(bins))]

# def discretize(sample, grid):
#     encodings = ()
#     for s in sample:
#         n_states = len(s)
#         encodings += (tuple(np.digitize(s[n],grid[n]) for n in range(n_states)),)
#     return encodings

# def tile_encode(sample, tilings):
#     encoded = [discretize(sample, grid) for grid in tilings]
#     return list(zip(encoded[0],encoded[1],encoded[2]))

# # low = [-1.0, -5.0]
# # high = [1.0, 5.0]
# # bins = [(10,10),(10,10),(10,10)]
# # offsets = [(-0.066, -0.33), (0,0),(0.066,0.33)]
# # samples = [(-1.2 , -5.1 ),
# #            (-0.75,  3.25),
# #            (-0.5 ,  0.0 ),
# #            ( 0.25, -1.9 ),
# #            ( 0.15, -1.75),
# #            ( 0.75,  2.5 ),
# #            ( 0.7 , -3.7 ),
# #            ( 1.0 ,  5.0 )]

# n_bins = 10
# bins = tuple([n_bins]*env.observation_space.shape[0])
# offsets = [(-0.066, -0.33), (0,0),(0.066,0.33)]

# exit()
# tilings = create_tilings(env.observation_space.low, env.observation_space.high, bins,offsets)
# print(tilings)
# exit()
# # tile_encode(samples, tilings)

# class QTable:
#     def __init__(self, state_size, action_size):
#         # Initializes a single Q Table with dimensions [state_size]
#         self.state_size = state_size
#         self.action_size = action_size
#         self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
#         # print("QTable(): size =", self.q_table.shape)

# class TiledQTable:
#     def __init__(self, tilings, action_size):
#         self.tilings = tilings

#         # basically state sizes = number of buckets + 1
#         self.state_sizes = [tuple(len(buckets)+1 for buckets in tiling_grid) for tiling_grid in self.tilings]
#         self.action_size = action_size
        
#         # create a q table for each bucket.
#         self.q_tables = [QTable(state_size, self.action_size) for state_size in self.state_sizes]

#     def get(self, state, action):
#         encoded_state = tile_encode(state, self.tilings)
#         # to get the action value, access, for each tiling_grid, the value in the q table, then average it out.
#         value = 0
#         for idx, q_table in zip(encoded_state[0], self.q_tables):
#             value += q_table.q_table[tuple(idx + (action,))]
#         value /= len(self.q_tables)
#         return value

#     def update(self, state, action, value, alpha=0.1):
#         encoded_state = tile_encode(state, self.tilings)

#         for idx, q_table in zip(encoded_state[0], self.q_tables):
#             target = q_table.q_table[tuple(idx + (action,))] # retrieving the target
#             q_table.q_table[tuple(idx + (action,))] = alpha * value + (1 - alpha) * target 

# # tiled_q = TiledQTable(tilings, 2)
# # tiled_q.get([[-1.2,-5.1]],1)

# class QAgentTileCoding:
#     def __init__(self, env, tiled_q, alpha=0.02, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1):
#         self.env = env
#         self.tiled_q = tiled_q
#         self.state_sizes = tiled_q.state_sizes
#         self.action_size = self.env.action_space.n

#         self.lr = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon

#     def reset_episode(self, state):
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
#         self.last_state = state

#         for action in range(self.action_size):
#             print(state)
#             print(self.tiled_q.get([state],action))
#         Q_s = [self.tiled_q.get([state], action) for action in range(self.action_size)]
#         self.last_action = np.argmax(Q_s)
#         return self.last_action

#     def reset_exploration(self, epsilon=None):
#         """Reset exploration rate used when training."""
#         self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

#     def act(self, state, reward=None, done=None, mode='train'):
#         """Pick next action and update internal Q table (when mode != 'test')."""
#         Q_s = [self.tiled_q.get([state], action) for action in range(self.action_size)]
#         # Pick the best action from Q table
#         greedy_action = np.argmax(Q_s)
#         if mode == 'test':
#             # Test mode: Simply produce an action
#             action = greedy_action
#         else:
#             # Train mode (default): Update Q table, pick next action
#             # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
#             value = reward + self.gamma * max(Q_s)
#             self.tiled_q.update(self.last_state, self.last_action, value, self.alpha)

#             # Exploration vs. exploitation
#             do_exploration = np.random.uniform(0, 1) < self.epsilon
#             if do_exploration:
#                 # Pick a random action
#                 action = np.random.randint(0, self.action_size)
#             else:
#                 # Pick the greedy action
#                 action = greedy_action

#         # Roll over current state, action for next step
#         self.last_state = state
#         self.last_action = action
#         return action

# agent = QAgentTileCoding(env, TiledQTable(tilings,2))

# def run(agent, env, num_episodes=10000, mode='train'):
#     """Run agent in given reinforcement learning environment and return scores."""
#     scores = []
#     max_avg_score = -np.inf
#     for i_episode in range(1, num_episodes+1):
#         # Initialize episode
#         state = env.reset()
#         action = agent.reset_episode(state)
#         total_reward = 0
#         done = False

#         # Roll out steps until done
#         while not done:
#             state, reward, done, info = env.step(action)
#             total_reward += reward
#             action = agent.act(state, reward, done, mode)

#         # Save final score
#         scores.append(total_reward)

#         # Print episode stats
#         if mode == 'train':
#             if len(scores) > 100:
#                 avg_score = np.mean(scores[-100:])
#                 if avg_score > max_avg_score:
#                     max_avg_score = avg_score
#             if i_episode % 100 == 0:
#                 print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
#                 sys.stdout.flush()
#     return scores

# scores = run(agent, env)