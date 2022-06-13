import numpy as np

import gym

import tensorflow as tf
from keras import layers, activations, Model, Sequential

EPOCHS = 20000
ALPHA = 0.8  # lr
GAMMA = 0.95  # discount rate

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001


