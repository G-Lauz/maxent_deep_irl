import torch


def deterministic_value_iteration(transition_probabilities, reward, gamma, epsilon, max_iteration=1000, save_value_functions=False):
    """
    Based on R. S. Sutton et A. G. Barto, Reinforcement learning: an introduction, Second edition. in Adaptive
    computation and machine learning series. Cambridge, Massachusetts: The MIT Press, 2018.
    and
    Based on RL Course by David Silver - Lecture 2: Markov Decision Process, (13 mai 2015). View on: Mai 22, 2024.
    [Online Video]. Available at: https://www.youtube.com/watch?v=lfHX2hHRMVQ

    :param transition_probabilities: A tensor of shape (n_states, n_states, n_actions) representing the transition probabilities of the MDP P(s' | s, a).
    :param reward: A tensor of shape (n_states) representing the reward of the MDP.
    :param gamma: The discount factor gamma in [0, 1].
    :param epsilon: A small threshold epsilon > 0 determining the accuracy of the estimation.
    :param max_iteration: The maximum number of iterations to perform.
    :param save_value_functions: A boolean indicating whether to save the value functions at each iteration.

    :return: The optimal value function of shape (n_states), the optimal policy of shape (n_states, n_actions) and the
        value functions at each iteration (None if save_value_functions is False).
    """
    n_states, _, n_actions = transition_probabilities.shape

    V = torch.zeros(n_states)

    value_functions = None
    if save_value_functions:
        value_functions = []

    for iteration in range(max_iteration):
        delta = 0

        Vt = V.clone()

        for s in range(n_states):
            V[s] = reward[s] + gamma * torch.max(transition_probabilities[:, s, :].T @ Vt)
            delta = max(delta, torch.abs(V[s] - Vt[s]))

        if save_value_functions:
            value_functions.append(V.clone())

        if delta < epsilon:
            break

    # Compute the optimal policy
    policy = torch.argmax(transition_probabilities.permute(2, 1, 0) @ V, dim=0)
    policy = torch.eye(n_actions)[policy]

    return V, policy, value_functions


class ValueIterationAgent:
    def __init__(self, mdp, gamma=0.99, epsilon=1E-5, save_value_functions=False):
        _, self.policy, self.value_functions = deterministic_value_iteration(
            mdp.transition_probabilities,
            mdp.reward,
            gamma,
            epsilon=epsilon,
            save_value_functions=save_value_functions
        )

    def act(self, state, jitter=None):
        best_action = torch.argmax(self.policy[state])

        if jitter is None:
            return best_action

        policy = self.policy[state].clone()
        policy[best_action] = policy[best_action] - jitter
        for action, probs in enumerate(policy):
            policy[action] = probs + (jitter / len(policy))

        probabilities = torch.distributions.Categorical(policy)
        return probabilities.sample()

    def get_value_functions(self):
        return self.value_functions

    def get_policy(self):
        return self.policy
