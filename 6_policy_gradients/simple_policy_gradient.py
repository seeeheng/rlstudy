# Just added comments / renamed stuff, but source code is from Spinning Up (OpenAI)
# https://github.com/openai/spinningup/

import torch
import torch.nn as nn
import gym
import numpy as np

def mlp(nn_shape, activation=nn.Tanh, output_activation=nn.Identity):
    """ Builds a feedforward neural network based on nn_shape.
    
    Args:
        nn_shape: [state dimensions, hiddenlayer1, ..., hiddenlayert, action dimensions]
        activation: activation function for hidden layers
        activation: activation function for output layer

    Returns:
        a MLP - nn.Sequential of a list of layers from input to output.
    """
    layers = []

    for i_current_layer in range(len(nn_shape)-1):
        current_activation = activation if i_current_layer < len(nn_shape)-2 else output_activation
        current_layer = nn_shape[i_current_layer]
        next_layer = nn_shape[i_current_layer + 1]
        layers += [nn.Linear(current_layer, next_layer), current_activation()]
    return nn.Sequential(*layers)

def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2, epochs=100, batch_size=5000, render=False):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, gym.spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "This example only works for envs with discrete action spaces."

    dim_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    logits_net = mlp([dim_observations] + hidden_sizes + [n_actions])

    # make function to compute action distribution
    def get_policy(observation):
        logits = logits_net(observation)
        return torch.distributions.Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(observation):
        # .item() is used on a torch tensor to extract the value of a tensor with a shape of (1,)
        return get_policy(observation).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(observation, action, weights):
        # for spg, we use the return from the episode as the weights 
        logp = get_policy(observation).log_prob(action)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = torch.optim.Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_observations = [] # for observations
        batch_actions = []      # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_returns = []      # for measuring episode returns
        batch_lengths = []      # for measuring episode lengths

        # reset episode-specific variables
        observation = env.reset()       # first observation comes from starting distribution
        done = False            # signal from environment that episode is over
        episode_rewards = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save observation
            batch_observations.append(observation.copy())

            # act in the environment
            action = get_action(torch.as_tensor(observation, dtype=torch.float32))
            observation, reward, done, _ = env.step(action)

            # save action, reward
            batch_actions.append(action)
            episode_rewards.append(reward)

            if done:
                # if episode is over, record info about episode
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lengths.append(episode_length)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [episode_return] * episode_length

                # reset episode-specific variables
                observation, done, episode_rewards = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # End experience loop if we have enough of it.
                # If not, just repeat to fill up the batch with experiences.
                if len(batch_observations) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            observation=torch.as_tensor(batch_observations, dtype=torch.float32),
            action=torch.as_tensor(batch_actions, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_returns, batch_lengths

    # training loop
    for i in range(epochs):
        batch_loss, batch_returns, batch_lengths = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t episode_length: %.3f'%
                (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)