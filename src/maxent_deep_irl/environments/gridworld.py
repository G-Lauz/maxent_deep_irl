"""
Windy Gridworld Environment from Sutton and Barto's book Reinforcement Learning: An Introduction, chapter 4.1., example 4.1., (page 72).
"""
import gymnasium
import pygame
import torch
import typing

from .base_mdp import BaseMDP


class WindyGridworld(gymnasium.Env, BaseMDP):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, grid_size: int, perturbation: torch.Tensor, render_mode=None, initial_state_distribution: typing.Union[int, torch.Tensor] = None, verbose=False):
        """
        :param grid_size: Size of the grid world
        :param perturbation: Perturbation matrix of shape (n_axis, grid_size). Each element represents a movement of a
            grid cell units in the given axis.
        :param render_mode: Rendering mode. Options: "human" for rendering with pygame, "rgb_array" for rendering with
            numpy array.
        :param initial_state_distribution: Initial state of the agent. If it is an integer, the agent starts from the given state.
            If it is an array of shape (n_states), the agent starts will be sample from the given distribution. If it is
            None, the agent starts will be sample from a uniform distribution.
        """
        self.window_size = 512  # Window size for rendering

        self.grid_size = grid_size
        self._n_states = grid_size ** 2
        self.observation_space = gymnasium.spaces.Discrete(self._n_states)

        self.actions = ((1, 0), (-1, 0), (0, -1), (0, 1), (0, 0))  # right, left, up, down, do nothing
        self._n_actions = len(self.actions)
        self.action_space = gymnasium.spaces.Discrete(4)

        if perturbation.shape != (2, grid_size):
            raise ValueError("Perturbation must have shape (2, grid_size)")
        self.perturbation = perturbation

        self._transition_probabilities = self.__build_transition_probabilities()

        if initial_state_distribution is None:
            self.initial_state_distribution = torch.distributions.Categorical(torch.ones(self._n_states) / self._n_states)
        elif isinstance(initial_state_distribution, int):
            self.initial_state_distribution = torch.distributions.Categorical(torch.eye(self._n_states)[initial_state_distribution])
        else:
            self.initial_state_distribution = torch.distributions.Categorical(initial_state_distribution)

        self.start_state = self.initial_state_distribution.sample().item()
        self.terminal_state = self._n_states - 1
        self.current_state = 0

        if render_mode is None or render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.verbose = verbose

        super().__init__()

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def reward(self) -> torch.Tensor:
        return torch.tensor([self._reward(state) for state in range(self._n_states)])

    @property
    def transition_probabilities(self) -> torch.Tensor:
        return self._transition_probabilities

    def get_next_position(self, coord, action):
        movement = torch.tensor(self.actions[action])
        next_position = coord + movement

        # Clip the position to the grid
        next_position[0] = torch.clamp(next_position[0], 0, self.grid_size - 1)
        next_position[1] = torch.clamp(next_position[1], 0, self.grid_size - 1)

        # Apply perturbation
        dx = self.perturbation[0, next_position[0].item()]
        dy = self.perturbation[1, next_position[1].item()]
        next_position[0] += int(dy)
        next_position[1] -= int(dx)

        # Clip the position to the grid
        next_position[0] = torch.clamp(next_position[0], 0, self.grid_size - 1)
        next_position[1] = torch.clamp(next_position[1], 0, self.grid_size - 1)

        return next_position


    def __build_transition_probabilities(self):
        transition_probabilities = torch.zeros(self._n_states, self._n_states, self._n_actions)  # P(s' | s, a)

        for state in range(self._n_states):
            coord = torch.tensor(self.__state_to_coordinate(state))

            for action in range(self._n_actions):
                next_position = self.get_next_position(coord, action)
                next_state = self.__coordinate_to_state(*next_position)
                transition_probabilities[next_state, state, action] = 1.0

        return transition_probabilities

    def _get_observation(self):
        return self.current_state

    def _get_info(self):
        return {}

    def __state_to_coordinate(self, state):
        return state % self.grid_size, state // self.grid_size

    def __coordinate_to_state(self, x, y):
        return int(y * self.grid_size + x)

    def __is_neighbor(self, x, y, x_prime, y_prime):
        return abs(x - x_prime) + abs(y - y_prime) == 1

    def __is_out_of_bounds(self, x, y):
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def _reward(self, state: int):
        if state == self._n_states - 1:
            return 0.0
        return -1.0

    def _is_terminal_state(self):
        return self.current_state == self.terminal_state

    def reset(self, state: int = None, seed=None, options=None):
        """
        Reset the environment to the initial state.

        :param state: Initial state of the agent. If it is None, the agent starts from the initial state distribution.
        :param seed: Seed for the random number generator.
        :param options: Options for the reset method.
        """
        super().reset(seed=seed)

        if state is not None:
            self.current_state = state
        else:
            self.current_state = self.initial_state_distribution.sample().item()

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if self.render_mode == "human":
            self._render_frame(action)

        probability_distribution = torch.distributions.Categorical(self._transition_probabilities[:, self.current_state, action])
        if self.verbose:
            print(f"P(s' | s={self.current_state}, a={action}) ->", end=" ")

        self.current_state = probability_distribution.sample().item()
        if self.verbose:
            print(f"s'={self.current_state}")

        terminated = self._is_terminal_state()
        reward = self._reward(self.current_state)
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, action=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        square_size = self.window_size // self.grid_size

        # Draw the perturbation
        perturbation_color_x = (255, 0, 0)
        perturbation_color_y = (255, 0, 0)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                shift_x = self.perturbation[0, x]
                shift_y = self.perturbation[1, y]

                if shift_x != 0 or shift_y != 0:
                    perturbation_x_surface = pygame.Surface((square_size, square_size))
                    perturbation_x_surface.fill((255, 255, 255, 0))
                    alpha_x = self._get_alpha(shift_x)
                    perturbation_x_surface.set_alpha(alpha_x)

                    direction = (square_size//2, 0) if shift_x > 0 else (square_size//2, square_size)
                    points = (direction, (0, square_size//2), (square_size, square_size//2))
                    pygame.draw.polygon(perturbation_x_surface, perturbation_color_x, points)
                    canvas.blit(perturbation_x_surface, (x * square_size, y * square_size))

                    perturbation_y_surface = pygame.Surface((square_size, square_size))
                    perturbation_y_surface.fill((255, 255, 255, 0))
                    alpha_y = self._get_alpha(shift_y)
                    perturbation_y_surface.set_alpha(alpha_y)

                    direction = (square_size, square_size//2) if shift_y > 0 else (0, square_size//2)
                    points = (direction, (square_size//2, 0), (square_size//2, square_size))
                    pygame.draw.polygon(perturbation_y_surface, perturbation_color_y, points)
                    canvas.blit(perturbation_y_surface, (x * square_size, y * square_size))

        # Draw the terminal state
        x, y = self.__state_to_coordinate(self._n_states - 1)
        pygame.draw.rect(canvas, (200, 200, 200), (x * square_size, y * square_size, square_size, square_size))

        # Draw state text overlay for each tile
        font = pygame.font.Font(None, 36)
        for state in range(self._n_states):
            x, y = self.__state_to_coordinate(state)
            text = font.render(str(state), True, (0, 0, 0))
            canvas.blit(text, (x * square_size + 10, y * square_size + 10))

        # Draw the target
        if action is not None:
            x, y = self.__state_to_coordinate(self.current_state)
            dx, dy = self.actions[action]
            x += dx
            y += dy
            pygame.draw.rect(canvas, (255, 0, 0), (x * square_size, y * square_size, square_size, square_size), 8, border_radius=1)

        # Draw the grid line
        for i in range(self.grid_size):
            pygame.draw.line(canvas, (0, 0, 0), (0, i * square_size), (self.window_size, i * square_size), width=3)
            pygame.draw.line(canvas, (0, 0, 0), (i * square_size, 0), (i * square_size, self.window_size), width=3)

        # Draw the agent
        x, y = self.__state_to_coordinate(self.current_state)
        x = (x + 0.5) * square_size
        y = (y + 0.5) * square_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (x, y),
            square_size // 3
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return pygame.surfarray.array3d(canvas).transpose(1, 0, 2)

    def _get_alpha(self, perturbation):
        max_intensity = torch.max(torch.abs(self.perturbation))
        alpha = int(torch.abs(perturbation) / max_intensity * 255)
        return alpha


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    GRID_SIZE = 4

    perturbation = torch.zeros(2, GRID_SIZE)
    perturbation[0, 1] = 1
    perturbation[1, 1] = -1

    env = WindyGridworld(GRID_SIZE, perturbation, initial_state_distribution=0, render_mode="human")

    # Validate transition probabilities distributions
    for state in range(env.n_states):
        for action in range(env.n_actions):
            assert torch.sum(env.transition_probabilities[:, state, action]) == 1.0, f"Transition probabilities P(s' | s={state}, a={action}) must sum to 1.0 : {env.transition_probabilities[:, state, action]}"

    env.reset()

    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()
