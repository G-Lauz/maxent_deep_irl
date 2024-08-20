import gymnasium

from .base_mdp import BaseMDP
from .toy_mdp import ToyMDP
from .gridworld import WindyGridworld


gymnasium.envs.registration.register(
    id="Toy-MDP-v0",
    entry_point="rl_algorithms.environments.toy_mdp:ToyMDP",
    max_episode_steps=30
)

gymnasium.envs.registration.register(
    id="Windy-Gridworld-v0",
    entry_point="rl_algorithms.environments.grid_world:WindyGridworld",
    max_episode_steps=300
)


__all__ = [
    "BaseMDP",
    "ToyMDP",
    "WindyGridworld"
]
