import gym
import numpy as np

env = gym.make("FrozenLake-v0")
env.reset()
env.render()
env.step(0)
env.render()

class GymEnvironment:
    def __init__(self, env_id):
        self.env = None
        self.n_states = None
        self.n_actions = None 
        self.make_environment(env_id)

    def make_environment(self,env_id):
        try:
            self.env = env.gym.make(env_id)
            self.env.reset()
            self.n_states = self.env.nS
            self.n_actions = self.env.nA
        except gym.error.Error as e:
            print("Error in making env. Could you have typed a wrong id? ID={}".format(env_id))
            print(e)
    
    def step(self,action):
        self.env.step(action)
    
    def reset(self):
        self.env.reset()