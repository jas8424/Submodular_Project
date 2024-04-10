import numpy as np
import random

def dist_gradient(A_to_B, B_to_A, vect_x, p=50):
    """Compute the gradient of the objective function."""
    A_length, B_length = len(A_to_B), len(B_to_A)
    gradient_martix = np.zeros((A_length, B_length))
    learning_rate = 1 / p

    for _ in range(p):
        sampled_Bs = np.nonzero(np.random.rand(B_length) <= vect_x)[0]
        for i, A_to_B_i in A_to_B.items():
            gradient_martix[i, np.setdiff1d(A_to_B_i, sampled_Bs, assume_unique=True)] += learning_rate

    return np.mean(gradient_martix, axis=0)


def dist_cont_greedy(A_to_B, B_to_A, k=10):
    vect_x = np.zeros(len(B_to_A))
    for _ in range(k):
        vect_v = dist_gradient(A_to_B, B_to_A, vect_x)
        vect_x += 1.0 / k * vect_v

    return np.argsort(-vect_x)[:k]
