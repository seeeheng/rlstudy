import torch.nn as nn

class Actor(nn.Module):
    def _distribution(self, observation):
        raise NotImplementedError

    def _log_prob_from_distribution(self, policy, action):
        raise NotImplementedError

    def forward(self, observation, action=None):
        """ Computes log probability given an action.

        Args:
            observation: current state
            action: action to be assessed
        Returns:
            policy: distribution
            logp_a: log prob given action
        """
        policy = self._distribution(observation)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)
        return policy, logp_a
