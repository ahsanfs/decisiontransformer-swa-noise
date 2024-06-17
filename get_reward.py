import gym
import numpy as np

import collections
import pickle

import d4rl


env_name = 'hopper'
dataset_type = 'medium'

name = f'{env_name}-{dataset_type}-v2'
env = gym.make(name)

print("Correct")

return_mean = 1442.1303638002685

print("Normalized Score:", env.get_normalized_score(return_mean))
