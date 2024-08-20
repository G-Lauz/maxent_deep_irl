import matplotlib.pyplot as plt
import numpy
import torch

from maxent_deep_irl.environments import WindyGridworld
from maxent_deep_irl import ValueIterationAgent


GRID_SIZE = 7


def plot_value_functions(value_functions):
    n_iterations = len(value_functions)
    n_states = len(value_functions[0])

    n_cols = 3
    n_rows = n_iterations // n_cols
    n_rows += n_iterations % n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, value_function in enumerate(value_functions):
        axes[i].imshow(value_function.reshape(GRID_SIZE, GRID_SIZE), cmap="hot", interpolation="nearest")
        axes[i].set_title(f"Value function at iteration {i}")
        axes[i].set_xticks(range(GRID_SIZE))
        axes[i].set_yticks(range(GRID_SIZE))

    plt.show()


def plot_policy(policy):
    action_symbols = ["→", "←", "↑", "↓"]
    policy = policy.argmax(dim=1).numpy().reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    checkerboard_pattern = numpy.indices((GRID_SIZE, GRID_SIZE)).sum(axis=0) % 2
    ax.imshow(checkerboard_pattern)

    ax.set_title("Policy")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ax.text(i, j, action_symbols[policy[j, i]], color="black", ha="center", va="center")

    plt.show()


def main():
    perturbation = torch.zeros(2, GRID_SIZE)
    perturbation[0, 1] = 1
    perturbation[1, 0] = -1

    perturbation[0, 2] = 1
    perturbation[0, 3] = 2
    perturbation[0, 4] = 1

    env = WindyGridworld(grid_size=GRID_SIZE, perturbation=perturbation, initial_state_distribution=0, render_mode="human")
    agent = ValueIterationAgent(env, gamma=0.99, epsilon=1E-5, save_value_functions=True)

    plot_value_functions(agent.get_value_functions())
    plot_policy(agent.get_policy())

    state, _ = env.reset()

    terminated = False
    while not terminated:
        action = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main() # pylint: disable=missing-kwoa
