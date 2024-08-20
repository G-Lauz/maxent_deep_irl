"""
Maximum Entropy Deep Inverse Reinforcement Learning (MaxEnt Deep IRL) implementation.

Based on the paper:
 - M. Wulfmeier, P. Ondruska, et I. Posner, "Maximum Entropy Deep Inverse Reinforcement Learning". arXiv, 11 mars 2016. doi: 10.48550/arXiv.1507.04888.
"""

import torch
import tqdm

from .environments import BaseMDP
from .value_iteration import deterministic_value_iteration


class MaximumEntropyDeepIRL:
    def __init__(self, env: BaseMDP, trajectories, reward_net: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 epochs: int, gamma: float, device="cpu", verbose: bool = False):
        """
        Initialize the Maximum Entropy Deep Inverse Reinforcement Learning (MaxEnt Deep IRL) algorithm.

        :param env: The environment as a subclass of BaseMDP.
        :param trajectories: The trajectories of the expert of shape (n_trajectories, variable number of step, 2) where
            the last dimension is the state-action pair.
        :param reward_net: The reward network as a torch.nn.Module that return a tensor of shape (n_states,).
        :param optimizer: The optimizer to use for training the reward network.
        :param epochs: The number of epochs to train the reward network, usually is equal to the number of trajectories.
        :param gamma: The discount factor gamma in [0, 1].
        :param device: The device to use.
        :param verbose: Whether to display the training progress.
        """
        self.env = env
        self.trajectories = trajectories
        self.reward_net = reward_net
        self.optimizer = optimizer
        self.epochs = epochs
        self.gamma = gamma
        self.device = device
        self.verbose = verbose

    def state_action_frequency(self):
        """
        Compute the state-action frequency from the given trajectory as shown in the section 3.1. Training Procedure of (Wulfmeir & al, 2016).
        """

        state_action_counts = torch.zeros(self.env.n_states, self.env.n_actions)
        for trajectory in self.trajectories:
            for state, action in trajectory:
                state_action_counts[state, action] += 1

        state_action_freq = state_action_counts.float() / len(self.trajectories)

        return state_action_freq

    def approximate_value_iteration(self, reward, epsilon=1E-5, max_iteration=1000):
        """
        Approximate the value iteration with soft-max value iteration as shown in the algorithm 2 of (Wulfmeir & al, 2016).

        :param reward: A tensor of shape (n_states,) representing the reward of the MDP.
        :param epsilon: A small threshold epsilon > 0 determining the accuracy of the estimation.
        :param max_iteration: The maximum number of iterations to perform.

        :return: The policy of shape (n_states, n_actions) and the value function of shape (n_states,).
        """
        V = torch.full((self.env.n_states,), -1E6)

        for iteration in range(max_iteration):
            Vt = V.clone()
            Vt[-1] = 0  # terminal state

            Q = torch.zeros((self.env.n_states, self.env.n_actions))
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    Q[s, a] = reward[s] + self.gamma * torch.sum(self.env.transition_probabilities[:, s, a] * Vt)

            V = torch.logsumexp(Q, dim=1)  # soft-max over actions

            if torch.max(torch.abs(V - Vt)) < epsilon:
                break

        policy = torch.exp(Q - V.reshape(-1, 1))

        return policy, V

    def propagate_policy(self, policy, epsilon=1E-6, max_iteration=1000):
        """
        Propagate the policy through the environment dynamics as shown in the algorithm 3 of (Wulfmeier & al, 2016).

        :param policy: The policy of shape (n_states, n_actions).
        :param epsilon: The early stopping threshold.
        :param max_iteration: The maximum number of iteration for the policy propagation

        :return: The expected states visitation frequency of shape (n_states,) according to the given policy
        """
        state_visitation_frequency = torch.zeros((len(self.trajectories[0]), self.env.n_states))

        initial_state_distribution = torch.zeros(self.env.n_states)
        for traj in self.trajectories:
            initial_state_distribution[int(traj[0][0])] += 1.0
        state_visitation_frequency[0] = initial_state_distribution / len(self.trajectories)

        for t in range(1, len(self.trajectories[0])):
            state_visitation_frequency[t] = torch.sum(torch.sum(self.env.transition_probabilities * policy * state_visitation_frequency[t - 1].reshape(-1, 1), dim=1), dim=1)
        return state_visitation_frequency.sum(dim=0)

    def train(self):
        """
        Train the reward network using the Maximum Entropy Deep Inverse Reinforcement Learning (MaxEnt Deep IRL) algorithm
        as shown in the algorithm 1 of (Wulfmeier & al, 2016).

        :return: The reward of shape (n_states,) and a dictionary of metrics accumulated during the training.
        """
        optimal_value_function, _, _ = deterministic_value_iteration(self.env.transition_probabilities, self.env.reward, self.gamma, epsilon=1E-5)

        state_action_freq = self.state_action_frequency()
        feature_expectation = state_action_freq.sum(dim=1)

        features = torch.eye(self.env.n_states)

        expected_value_differences = []
        losses = []
        policies = []
        rewards = []
        learned_value_function = None

        for epoch in tqdm.tqdm(range(self.epochs), disable=not self.verbose):
            reward_estimation = self.reward_net(features).flatten()

            policy, learned_value_function = self.approximate_value_iteration(reward_estimation.detach())
            state_visitation_frequency = self.propagate_policy(policy)

            expected_value_difference = torch.abs(optimal_value_function - learned_value_function)

            loss = -torch.sum(torch.log(policy + 1E-6) * state_action_freq, dim=0)
            loss_gradient = feature_expectation - state_visitation_frequency

            self.optimizer.zero_grad()
            reward_estimation.backward(-loss_gradient)
            self.optimizer.step()

            expected_value_differences.append(expected_value_difference.detach().numpy())
            losses.append(loss.detach().numpy())
            policies.append(policy.detach().numpy())
            rewards.append(reward_estimation.detach().numpy())

        metrics = {
            "expected_value_differences": expected_value_differences,
            "value_functions": learned_value_function,
            "losses": losses,
            "policies": policies,
            "rewards": rewards
        }

        reward = self.reward_net(features).flatten()
        return reward, metrics
