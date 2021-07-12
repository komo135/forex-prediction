import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

x, y = np.load("x.npy"), np.load("y.npy")
