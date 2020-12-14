import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

# constants
BUFFER_LENGTH=2000
EPSILON_DECAY=0.995

## CARTPOLE Notes
# Epsilon decay @ 0.9 = no convergence 
# Epsilon decay @ 0.995 = slow convergence
# Epsilon decay @ 0.99 = fast!
# LR @ 0.0005 = slow convergence
# LR @ 0.005 = way faster. like < 100 episodes
# LR @ 0.05 = way way faster.

GAMMA = 0.9
BATCH_SIZE = 128
LR = 0.1
N_EPISODES = 2000
PRINT_EVERY = 1
TAU = 0.08
LEARN_EVERY = 4
MAX_T = 200000

env_name = "MountainCar-v0"
# env_name = "CartPole-v0"
env = gym.make(env_name)

# N_STATES = 4
# N_ACTIONS = 2

#ACROBOT
# N_STATES = 2
# N_ACTIONS = 3

#MC
N_STATES = 2
N_ACTIONS = 3

# print(env.reset())
# exit()

"""
(From mnih et. al)
1. Store agent's experiences at each time step; e_t = (s_t, a_t, r_t, s_t+1)
2. During inner loop of the algorithm, apply Q-learning updates (minibatch updates)
to samples of experiences, drawn at random from the pool of stored samples.
3. Work on fixed length representations of histories.
4. Algorithms only store N before releasing it. (probably good to use deque)
"""
class ExperienceReplay:
    def __init__(self):
        self.memory = deque(maxlen=BUFFER_LENGTH)

    def add_memory(self, state, action, reward, done, next_state):
        # TODO: check the data type getting fed into here
        experience = (state, action, reward, done, next_state)
        self.memory.append(experience)

    def retrieve_memory(self):
        return random.sample(self.memory,BATCH_SIZE)

"""
(From mnih et. al)
1. 256 neurons in hidden layer, one hidden layer for decision making.
2. Epsilon is decayed from 1 to 0.1 linearly over 1 million frames.
    Since I'm not training end-to-end, I'll create a faster decay.
3. Target is normal TD target (i.e. reward + gamma(max_a(Q(next_state))))
    Target is only reward if state is terminal.
"""
class DQN(tf.keras.Model):
    def __init__(self,n_actions):
        super(DQN, self).__init__()
        # self.layer1 = tf.keras.layers.Dense(256, input_shape=(6,), activation='relu')
        self.layer1 = tf.keras.layers.Dense(128, activation='relu')
        # self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.layer2 = tf.keras.layers.Dense(n_actions) # TODO, actions
    
    def call(self, x, training=True):
        x = self.layer1(x)
        # if training:
            # x = self.dropout1(x, training=training)
        x = self.layer2(x)
        return x

class DqnAgent:
    def __init__(self, n_states, n_actions, env, epsilon=1, lr=LR, gamma=GAMMA, buffer_length=BUFFER_LENGTH, batch_size=BATCH_SIZE):
        self.n_states = n_states
        self.n_actions = n_actions
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer_length = buffer_length
        self.batch_size = BATCH_SIZE
        self.lr = lr

        self.primary_dqn = DQN(self.n_actions)
        self.target_dqn = DQN(self.n_actions) # TODO: implement DD for more stability

        self.replay = ExperienceReplay()

        self.current_state = None
        self.episode_reward = None
        self.total_reward = 0
        self.t_since_learn = 1
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _train_step(self, targets, old_states, actions):
        with tf.GradientTape() as tape:
            predictions = self.primary_dqn(old_states, training=True)
            
            # Some preproc for action_indices so it will work
            action_indices = actions
            row_indices = tf.range(tf.shape(action_indices)[0])
            full_indices = tf.stack([row_indices, action_indices],axis=1)               
            predictions = tf.gather_nd(predictions,full_indices)
            loss = self.loss_object(targets,predictions)
        gradients = tape.gradient(loss, self.primary_dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.primary_dqn.trainable_variables))
        return loss

    def first_step(self, current_state):
        self.current_state = current_state
        self.episode_reward = 0

    def step(self):
        current_action = self.act(self.current_state)
        next_state, reward, done, _ = self.env.step(current_action)
        self.replay.add_memory(self.current_state, current_action, reward, done, next_state)
        # After every self.batch_size timestep, learn.
        if (self.t_since_learn % LEARN_EVERY) == 0:
            self.learn()            
        self.current_state = next_state
        self.episode_reward += reward
        if done:
            self.epsilon = max(self.epsilon*EPSILON_DECAY,0.1)
            self.total_reward += self.episode_reward
        self.t_since_learn += 1
        return done

    def act(self, current_state):
        # in this particular example, I'm building a deterministic policy with epsilon implementation.
        current_state = current_state.reshape(1,-1) # turning it into a "batch" which keras can work with
        current_argmax = np.argmax(self.primary_dqn(current_state,training=False).numpy())
        probs = np.ones(self.n_actions) * (self.epsilon/self.n_actions)
        probs[current_argmax] += 1-self.epsilon 
        action = np.random.choice(np.arange(self.n_actions),p=probs)
        return action

    def learn(self):
        if len(self.replay.memory) < BATCH_SIZE:
            return

        experiences = self.replay.retrieve_memory()
        # experiences are returned in the form [ [[old_state],action, reward, done, [new_state]], ... blah blah blah ]
        """
        if not done:
            TARGET = reward + gamma * max Q(next_state) 
        else:
            TARGET = reward
        """ 
        old_states  = np.array([e[0] for e in experiences])
        actions     = np.array([e[1] for e in experiences])
        rewards     = np.array([e[2] for e in experiences])
        dones       = np.array([e[3] for e in experiences])
        new_states  = np.array([e[4] for e in experiences])    
        # targets = []

        # Building the targets. Begin with Q(s,a)
        
        # targets = self.primary_dqn(new_states,training=False).numpy().max(1) # getting the maximum values of each Q(new_state)
        targets = tf.reduce_max(self.target_dqn(new_states,training=False), axis=1)
        targets = rewards + (self.gamma * targets * (1-dones)) # completing target: r + gamma * max(Q_new_state) * (1-done) so = R if done, target otherwise.
        loss = self._train_step(targets, old_states, actions)

        # TODO: once your q network actually learns.
        self.update()

    def update(self):
        # self.target_dqn.set_weights(self.primary_dqn.get_weights()) 
        # primary_weights = self.primary_dqn.get_weights()
        # target_weights = self.target_dqn.get_weights()
        for t, e in zip(self.target_dqn.trainable_variables, self.primary_dqn.trainable_variables): 
            t.assign(t * (1 - TAU) + e * TAU)

dqn_agent = DqnAgent(N_STATES,N_ACTIONS,env)
current_state = env.reset()
scores_window = deque(maxlen=100)
average_reward_plot = []
best_average_reward = -np.inf

for i_episode in range(N_EPISODES):
    done = False
    dqn_agent.first_step(current_state)
    t = 0
    while not done:
        t+=1
        if t > MAX_T:
            done = True
            break
        done = dqn_agent.step()
    current_state = dqn_agent.env.reset()

    scores_window.append(dqn_agent.episode_reward)
    running_average = np.mean(scores_window)
    if i_episode % PRINT_EVERY == 0:
        average_reward_plot += [[i_episode, running_average],]        
        if running_average > best_average_reward:
            best_average_reward = running_average
        print("Episode {}/{}, best_average_reward={:.2f}, running_average={:.2f}\r".format(i_episode,N_EPISODES,best_average_reward,running_average), end="")
    if i_episode % 100 == 0:
        print("Episode {}/{}, best_average_reward={:.2f}, running_average={:.2f}\r".format(i_episode,N_EPISODES,best_average_reward,running_average))        

print("\n um, done.")
dqn_agent.primary_dqn.save_weights("./checkpoints/dqn-{}-2".format(env_name))

"""
ACROBOT done and trained here with hyperparameters:
GAMMA = 0.9
BATCH_SIZE = 128
LR = 0.005
N_EPISODES = 1000
PRINT_EVERY = 1
TAU = 0.08
LEARN_EVERY = 4
MAX_T = 2000

Episode 0/1000, best_average_reward=-500.00, running_average=-500.00
Episode 100/1000, best_average_reward=-463.63, running_average=-463.63
Episode 200/1000, best_average_reward=-359.63, running_average=-360.21
Episode 300/1000, best_average_reward=-281.26, running_average=-281.26
Episode 400/1000, best_average_reward=-228.54, running_average=-236.28
Episode 500/1000, best_average_reward=-217.61, running_average=-218.28
Episode 600/1000, best_average_reward=-173.22, running_average=-179.04
Episode 700/1000, best_average_reward=-173.22, running_average=-195.61
Episode 800/1000, best_average_reward=-173.22, running_average=-201.65
Episode 900/1000, best_average_reward=-173.22, running_average=-211.99
Episode 999/1000, best_average_reward=-173.22, running_average=-194.49
 um, done.
"""