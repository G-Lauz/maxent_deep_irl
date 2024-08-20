import abc

import torch


class BaseMDP(abc.ABC):
    _n_states: int
    _n_actions: int
    reward: torch.Tensor
    _transition_probabilities: torch.Tensor
    initial_state_distribution: torch.distributions.Categorical

    def __init__(self):
        assert self.reward.shape == (self.n_states,), f"Reward shape must be (n_states,) but got {self.reward.shape}"
        assert self.transition_probabilities.shape == (self.n_states, self.n_states, self.n_actions), f"Transition probabilities P(s' | s, a) shape must be (n_states, n_states, n_actions) but got {self.transition_probabilities.shape}"

    @property
    @abc.abstractmethod
    def n_states(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def transition_probabilities(self) -> torch.Tensor:
        raise NotImplementedError
