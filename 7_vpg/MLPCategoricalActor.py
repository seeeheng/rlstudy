import torch
from Actor import Actor
from utils.rlutils import mlp

class MLPCategoricalActor(Actor):
    def __init__(self, dim_observations, dim_actions, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([dim_observations] + hidden_sizes + [dim_actions], activation=activation)

    def _distribution(self, observation):
        """ Utilizes in-built categorical distribution for discrete action spaces. """
        logits = self.logits_net(observation)
        return torch.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, policy, action):
        """ Gets log_prob from categorical distribution for action. """
        return policy.log_prob(action)
