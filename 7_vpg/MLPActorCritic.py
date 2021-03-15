import torch.nn as nn
from gym.spaces import Box, Discrete
from MLPCritic import MLPCritic

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        dim_observations = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.policy = MLPGaussianActor(dim_observations, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.policy = MLPCategoricalActor(dim_observations, action_space.n, hidden_sizes, activation)

        self.value_function = MLPCritic(dim_observations, hidden_sizes, activation)

    def step(self, observation):
        with torch.no_grad():
            policy_distribution = self.policy._distribution(observation)
            action = policy_distribution.sample()
            logp_a = self.policy._log_prob_from_distribution(policy_distribution, action)
            value = self.value_function(observation)
        return action.numpy(), value.numpy(), logp_a.numpy()

    def act(self, observation):
        action, _, _ = self.step(observation)
        return action

