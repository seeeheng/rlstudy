import torch.nn as nn
from utils.rlutils import mlp

class MLPCritic(nn.Module):
	def __init__(self, dim_observations, hidden_sizes, activation):
		super().__init__()
		self.v_net = mlp([dim_observations] + hidden_sizes, activation=activation)

	def forward(self, observation):
		return torch.squeeze(self.v_net(observation), -1)
