import matplotlib.pyplot as plt
import numpy
import torch
import tqdm

from maxent_deep_irl.environments import WindyGridworld
from maxent_deep_irl import MaximumEntropyDeepIRL, ValueIterationAgent


def set_seed(seed):
    """
    Define the seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    numpy.random.seed(seed)


def generate_demonstrations(agent, mdp, n_trajectories, n_steps, jitter=0, seed=0, full_coverage: bool = False):
    mdp.action_space.seed(seed)
    mdp.observation_space.seed(seed)

    if full_coverage:
        n_trajectories = mdp.n_states

    demonstrations = []
    for i in tqdm.tqdm(range(n_trajectories)):
        trajectory = []

        overrided_state = i if full_coverage else None
        state, _ = mdp.reset(state=overrided_state, seed=seed)

        # done = False
        # while not done:
        for i in range(n_steps):
            action = agent.act(state, jitter=jitter).item()
            trajectory.append((state, action))
            state, reward, terminated, truncated, info = mdp.step(action)

            # done = terminated or truncated

        # trajectory.append((state, mdp.n_actions - 1))  # Add the terminal state-action

        demonstrations.append(trajectory)

    return demonstrations


def plot_value_function(value_function, prefix=""):
    plt.figure()
    plt.imshow(value_function.reshape(GRID_SIZE, GRID_SIZE), cmap="hot", interpolation="nearest")
    plt.title(f"{prefix}Value function")
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))


def plot_policy(policy, prefix=""):
    action_symbols = ["→", "←", "↑", "↓", "x"]
    policy = policy.argmax(axis=1).reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    checkerboard_pattern = numpy.indices((GRID_SIZE, GRID_SIZE)).sum(axis=0) % 2
    ax.imshow(checkerboard_pattern)

    ax.set_title(f"{prefix} - Policy")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ax.text(i, j, action_symbols[policy[j, i]], color="black", ha="center", va="center")


def plot_reward(reward, mdp, prefix=""):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(mdp.reward.reshape(GRID_SIZE, GRID_SIZE), cmap="hot", interpolation="nearest")
    axes[0].set_title(f"{prefix}Real rewards")
    axes[0].set_xticks(range(GRID_SIZE))
    axes[0].set_yticks(range(GRID_SIZE))

    axes[1].imshow(reward.reshape(GRID_SIZE, GRID_SIZE), cmap="hot", interpolation="nearest")
    axes[1].set_title(f"{prefix}Estimated rewards")
    axes[1].set_xticks(range(GRID_SIZE))
    axes[1].set_yticks(range(GRID_SIZE))


def plot_metrics(metrics, mdp, prefix=""):
    expected_value_differences = numpy.array(metrics["expected_value_differences"])
    losses = numpy.array(metrics["losses"])
    policies = metrics["policies"]
    rewards = numpy.array(metrics["rewards"])
    value_functions = metrics["value_functions"]

    plt.figure()
    plt.plot(expected_value_differences)
    plt.xlabel("Epoch")
    plt.ylabel("Expected value difference")
    plt.title(f"{prefix} - Expected value difference")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} - Loss")

    plt.figure()
    for state in range(mdp.n_states):
        plt.plot(rewards[:, state], label=f"State {state}")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title(f"{prefix} - Reward")
    plt.legend()

    plot_value_function(value_functions, prefix=prefix)
    plot_policy(policies[-1], prefix=prefix)
    plot_reward(rewards[-1], mdp, prefix=prefix)

    plt.show()


GRID_SIZE = 7
SEED = 0

def main():
    set_seed(SEED)

    discount = 0.90

    n_trajectories = 100
    n_steps = 30

    perturbation = torch.zeros(2, GRID_SIZE)
    perturbation[0, 1] = 1
    perturbation[1, 0] = -1

    perturbation[0, 2] = 1
    perturbation[0, 3] = 2
    perturbation[0, 4] = 1

    mdp = WindyGridworld(grid_size=GRID_SIZE, perturbation=perturbation, initial_state_distribution=None)

    reward_net = torch.nn.Sequential(
        torch.nn.Linear(mdp.n_states, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        # torch.nn.Tanh()
    )
    optimizer = torch.optim.Adagrad(reward_net.parameters(), lr=0.05)

    expert_agent = ValueIterationAgent(mdp, gamma=discount, epsilon=1E-5)

    print("Generating demonstrations...")
    demonstrations = generate_demonstrations(expert_agent, mdp, n_trajectories, n_steps=n_steps, jitter=0, seed=SEED, full_coverage=False)
    print(f"{len(demonstrations)} demonstrations generated\n")

    print("Training the reward network...")
    maxent_deep_irl = MaximumEntropyDeepIRL(mdp, demonstrations, reward_net, optimizer, epochs=n_trajectories, gamma=discount, verbose=True)
    rewards, metrics = maxent_deep_irl.train()

    print("-" * 80)
    print(f"Real reward: {mdp.reward}")
    print(f"Estimated reward: {rewards}")

    plot_metrics(metrics, mdp)


if __name__ == "__main__":
    main()  # pylint: disable=missing-kwoa
