import os
import sys

import torch
from torch import nn

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

def normalize(X):
    for i in range(X.shape[0]):
       sample = X[i]  
       mean = np.mean(sample)
       std = np.std(sample)
       X[i] = (sample - mean) / std
    return X