import torch
from Actor import Actor
from utils.rlutils import mlp
import numpy as np

class MLPGaussianActor(Actor):
    def __init__(self, dim_observations, dim_actions, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp([dim_observations] + hidden_sizes + [dim_actions], activation=activation)
        log_std = -0.5 * np.ones(dim_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, observation):
        """ Utilizes Gaussian for continuous action spaces. """
        mu = self.mu_net(observation)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def _log_prob_from_distribution(self, policy, action):
        """ Gets log_prob from Gaussian distribution for action. """
        return policy.log_prob(action).sum(axis=-1)
