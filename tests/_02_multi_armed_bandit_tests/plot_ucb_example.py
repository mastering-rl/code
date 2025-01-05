import math
import matplotlib.pyplot as plt

arm_pulls = [4000, 180, 100]
q_values = [0.65, 0.55, 0.4]

# Total number of pulls
t = sum(arm_pulls)

# calculate the UCB1 confidence interval for each arm
confidence_intervals = [math.sqrt(2 * math.log(t) / n) for n in arm_pulls]
ucb_values = [q + ci for q, ci in zip(q_values, confidence_intervals)]

# Plotting the results
arms = ["Arm 1", "Arm 2", "Arm 3"]
x_pos = range(len(arms))

# Bar graph
plt.bar(
    x_pos,
    q_values,
    yerr=confidence_intervals,
    capsize=10,
    alpha=0.7,
    width=0.5,
    color="skyblue",
)
plt.xticks(x_pos, arms)
plt.ylabel("Mean Reward")

# Plot UCB1 values as points above the bars
for i, ucb in enumerate(ucb_values):
    plt.plot(i, ucb, "ro")  # Red dot indicating UCB value

plt.show()
