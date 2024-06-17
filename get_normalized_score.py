import gym
import numpy as np

# Import the environment
import d4rl

# Setup the environment
env_name = 'hopper'
dataset_type = 'medium-expert'
name = f'{env_name}-{dataset_type}-v2'
score = 3600
print(name, score)
env = gym.make(name)

print("Environment setup is correct, added noise")

returns = [
    1254.6546480941965, 2173.935457069292, 3233.5067710782855,
    3422.3614230611565, 3511.940829109872, 3325.637446001193,
    3511.922888755066, 3489.038086656555, 3366.3785505169776,
    3174.243778825962
]

# Calculate normalized scores and store them
normalized_scores = []
for return_mean in returns:
    normalized_score = env.get_normalized_score(return_mean) * 100
    normalized_scores.append(normalized_score)
    print("Normalized Score:", normalized_score)

# Calculate the average and standard deviation of normalized scores
average_normalized_score = np.mean(normalized_scores)
std_dev_normalized_score = np.std(normalized_scores)

# Print the average ± standard deviation
print(f"Average Normalized Score: {average_normalized_score:.2f} ± {std_dev_normalized_score:.2f}")
