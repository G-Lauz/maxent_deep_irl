import matplotlib.pyplot as plt
import numpy
import torch
import tqdm

from maxent_deep_irl.environments import ToyMDP
from maxent_deep_irl import MaximumEntropyDeepIRL, ValueIterationAgent


def generate_demonstrations(agent, mdp, n_trajectories, seed=0):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    mdp.action_space.seed(seed)
    mdp.observation_space.seed(seed)
    demonstrations = []

    for _ in tqdm.tqdm(range(n_trajectories)):
        trajectory = []

        state, _ = mdp.reset(seed=seed)

        done = False
        while not done:
            action = agent.act(state).item()
            trajectory.append((state, action))
            state, reward, terminated, truncated, info = mdp.step(action)

            done = terminated or truncated

        demonstrations.append(trajectory)

    return torch.tensor(demonstrations)


def plot_value_function(value_function):
    plt.figure()
    plt.imshow(value_function.reshape(1, 2), cmap="hot", interpolation="nearest")
    plt.title("Value function")
    plt.xticks(range(2))
    plt.yticks(range(1))


def plot_policy(policy):
    action_symbols = ["Stay", "Switch"]
    policy = policy.argmax(axis=1).reshape(1, 2)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    checkerboard_pattern = numpy.indices((1, 2)).sum(axis=0) % 2
    ax.imshow(checkerboard_pattern, cmap="Wistia")

    ax.set_title("Policy")
    ax.set_xticks(range(2))
    ax.set_yticks(range(1))

    for i in range(2):
        ax.text(i, 0, action_symbols[policy[0, i]], color="black", ha="center", va="center")


def plot_reward(reward, mdp):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(mdp.reward.reshape(1, 2), cmap="hot", interpolation="nearest")
    axes[0].set_title("Real reward")
    axes[0].set_xticks(range(2))
    axes[0].set_yticks(range(1))

    axes[1].imshow(reward.reshape(1, 2), cmap="hot", interpolation="nearest")
    axes[1].set_title("Estimated reward")
    axes[1].set_xticks(range(2))
    axes[1].set_yticks(range(1))


def plot_metrics(metrics, mdp):
    expected_value_differences = numpy.array(metrics["expected_value_differences"])
    losses = numpy.array(metrics["losses"])
    policies = metrics["policies"]
    rewards = numpy.array(metrics["rewards"])
    value_functions = metrics["value_functions"]

    plt.figure()
    plt.plot(expected_value_differences)
    plt.xlabel("Epoch")
    plt.ylabel("Expected value difference")
    plt.title("Expected value difference")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")

    plt.figure()
    for state in range(mdp.n_states):
        plt.plot(rewards[:, state], label=f"State {state}")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Reward")
    plt.legend()

    plot_value_function(value_functions)
    plot_policy(policies[-1])
    plot_reward(rewards[-1], mdp)

    plt.show()


def main():
    mdp = ToyMDP(initial_state_distribution=0)

    expert_agent = ValueIterationAgent(mdp)

    print(f"Generating demonstrations...")
    demonstrations = generate_demonstrations(expert_agent, mdp, n_trajectories=100, seed=0)
    print(f"Generated {len(demonstrations)} demonstrations\n")

    reward_net = torch.nn.Sequential(
        torch.nn.Linear(mdp.n_states, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        # torch.nn.Tanh()
    )
    optimizer = torch.optim.Adagrad(reward_net.parameters(), lr=1E-1)

    # Initialize the reward net weights and biases
    for layer in reward_net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    print(f"Training reward estimator...")
    maxent_deep_irl = MaximumEntropyDeepIRL(mdp, demonstrations, reward_net, optimizer, epochs=100, gamma=0.99, verbose=True)
    rewards, metrics = maxent_deep_irl.train()

    print("-----------------------------")
    print(f"Real reward: {mdp.reward}")
    print(f"Estimated reward: {rewards}")

    plot_metrics(metrics, mdp)


if __name__ == "__main__":
    main()  # pylint: disable=missing-kwoa
