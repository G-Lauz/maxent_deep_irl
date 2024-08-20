"""
A toy MDP environment with two states and two actions for testing purposes.

Based on Markov Decision Processes, (February 12, 2015). Seen: May 22, 2024. [Online video]. Available at: https://www.youtube.com/watch?v=KovN7WKI9Y0
"""

import typing

import gymnasium
import pygame
import torch

from .base_mdp import BaseMDP


class ToyMDP(gymnasium.Env, BaseMDP):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, initial_state_distribution: typing.Union[int, torch.Tensor] = None, render_mode=None, verbose=False):
        self.gamma = 0.5

        self.optimal_value_function = torch.tensor(
            [4.4, 1.2]
        )

        self.optimal_policy = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        if initial_state_distribution is None:
            self._initial_state_distribution = torch.distributions.Categorical(torch.ones(self.n_states) / self.n_states)
        elif isinstance(initial_state_distribution, int):
            self._initial_state_distribution = torch.distributions.Categorical(torch.eye(self.n_states)[initial_state_distribution])
        else:
            self._initial_state_distribution = torch.distributions.Categorical(initial_state_distribution)

        self.current_state = None
        self.next_state = None

        self.verbose = verbose

        # Gymnasium setup
        self.window_size = 512

        self.observation_space = gymnasium.spaces.Discrete(self.n_states)
        self.action_space = gymnasium.spaces.Discrete(self.n_actions)

        self.truncation_limit = 30
        self.number_of_steps = 0

        if render_mode is None or render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode

        self.window = None
        self.clock = None

        super().__init__()


    @property
    def n_states(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def reward(self) -> torch.Tensor:
        return torch.tensor(
            [3.0, -1.0]
        )

    @property
    def transition_probabilities(self) -> torch.Tensor:
        return torch.tensor([
            [ # State 0 (s')
                # State 0, State 1 (s)
                [0.5, 0.0], # Action 0
                [0.0, 1.0]  # Action 1
            ],
            [ # State 1 (from)
                # State 0, State 1 (to)
                [0.5, 1.0], # Action 0
                [1.0, 0.0]  # Action 1
            ]
        ])

    @property
    def initial_state_distribution(self) -> torch.distributions.Categorical:
        return self._initial_state_distribution

    def _get_observation(self):
        return self.current_state

    def _get_reward(self, state):
        return self.reward[state]

    def _get_info(self):
        return {}

    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.number_of_steps >= self.truncation_limit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.number_of_steps = 0
        self.current_state = self.initial_state_distribution.sample().item()

        obersevation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obersevation, info

    def step(self, action, deterministic=False):
        if self.render_mode == "human":
            self._render_frame(action)

        self.number_of_steps += 1

        if deterministic:
            next_state = torch.argmax(self.transition_probabilities[:, self.current_state, action]).item()
        else:
            probability_distribution = torch.distributions.Categorical(self.transition_probabilities[:, self.current_state, action])
            next_state = probability_distribution.sample().item()

        if self.verbose:
            print(f"P(s' | s={self.current_state}, a={action}) -> s'={next_state}")

        self.current_state = next_state

        observation = self._get_observation()
        reward = self._get_reward(self.current_state)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()

    def _render_frame(self, action=None):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size // 2))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size // 2))
        canvas.fill((255, 255, 255))
        square_size = self.window_size // 2

        # Draw the intended action
        if action is not None:
            target_state = torch.argmax(self.transition_probabilities[:, self.current_state, action]).item()
            pygame.draw.rect(canvas, (255, 0, 0), (target_state * square_size, 0, square_size, square_size), 8, border_radius=1)

        # Draw state text overlay
        font = pygame.font.Font(None, 36)
        for state in range(self.n_states):
            text = font.render(f"State {state}", True, (0, 0, 0))
            canvas.blit(text, (state * square_size, 0))

        # Draw the grid line
        for i in range(1, self.n_states):
            pygame.draw.line(canvas, (0, 0, 0), (i * square_size, 0), (i * square_size, self.window_size), 2)

        # Draw the agent
        pygame.draw.circle(canvas, (0, 0, 255), (self.current_state * square_size + square_size // 2, square_size // 2), square_size // 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return pygame.surfarray.array3d(canvas).transpose(1, 0, 2)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    env = ToyMDP(render_mode="human", verbose=True)
    env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        env.render()

    env.close()
