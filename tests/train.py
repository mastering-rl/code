import numpy as np
import matplotlib.pyplot as plt
from mastering_rl.tests.plot import Plot


def train(
    mdp,
    policy,
    policy_name,
    learner,
    learner_name="",
    test=False,  # Whether to test the policy during training
    plot=True,   # Whether to plot the rewards during training
    max_episode_length=float('inf'),
    epochs=200,
    epoch_size=20,
):
    plt.ion()  # Turn on interactive mode for real-time plotting
    best_reward = float("-inf")
    train_rewards = []
    test_rewards = []
    for epoch in range(1, epochs + 1):
        train_rewards += learner.execute(epoch_size, max_episode_length)

        all_rewards = [train_rewards]
        labels = [learner_name + " train rewards"]
        log_message = f"| Epoch: {epoch} | Train reward: {np.mean(train_rewards[-epoch_size:])} |"
        policy_rewards = train_rewards
        if test:
            policy.set_stochastic(False)
            test_rewards += mdp.execute_policy(policy, episodes=epoch_size, max_episode_length=max_episode_length)
            policy.set_stochastic(True)
            all_rewards.append(test_rewards)
            labels.append(learner_name + " test rewards")
            log_message += f" | Test reward: {np.mean(test_rewards[-epoch_size:])} |"
            policy_rewards = test_rewards
        print(log_message)

        # If this is the best reward so far, save the policy
        if np.mean(policy_rewards[-epoch_size:]) > best_reward:
            policy.save("policies/" + policy_name)
            best_reward = np.mean(policy_rewards[-epoch_size:])
            print(f"Saving {policy_name}")

        if plot:
            plt.clf()  # Clear the current figure
            Plot.plot_cumulative_rewards(
                labels, all_rewards, smoothing_factor=0.0
            )
            plt.draw()
            plt.pause(0.1)

    plt.clf()
    plt.ioff()  # Turn of interactive mode
    return train_rewards, test_rewards