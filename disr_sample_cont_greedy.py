import numpy as np
import random

def dist_sample_gradient(A_to_B, B_to_A, vect_x, p=50):
    """
    Args:
        A_to_B (dict): Mapping from entities in A to sets of entities in set B.
        B_to_A (dict): Mapping from entities in B to sets of entities in set A.
        vect_x (numpy.ndarray): A numpy array representing the current state of the vector x.
        p (int, optional): Number of iterations for the gradient computation. Defaults to 50.

    Returns:
        numpy.ndarray: The gradient vector of the objective function.
    """
    A_length, B_length = len(A_to_B), len(B_to_A)
    sampled_A = A_length // 10
    learning_rate = 1 / p
    A_to_B_items = list(A_to_B.items())
    # Initialize a matrix to hold the gradient values.
    gradient_martix = np.zeros((A_length, B_length))

    for _ in range(p):
        # Sample Bs based on the probability distribution vect_x.
        sampled_Bs = np.nonzero(np.random.rand(B_length) <= vect_x)[0]
        # Randomly sample a subset of A_to_B to use in this iteration.
        sampled_A_to_B_items = random.sample(list(A_to_B_items), sampled_A)

        # For loop used to compute gradient, the logic looks weird but is correct and efficient hopefully
        for i, A_to_B_i in sampled_A_to_B_items:
            # Increase gradient for elements in A_to_B_i that are not in sampled_Bs.
            gradient_martix[i, np.setdiff1d(A_to_B_i, sampled_Bs, assume_unique=True)] += learning_rate

        '''
        Same as above for loop 
        for k in A_to_B_i:
            if k not in sampled_venues:
                gradient_list[i, k] += learning_rate  
        '''

    return np.sum(gradient_martix, axis=0) / sampled_A


def dist_sample_cont_greedy(A_to_B, B_to_A, k=10):
    """
    Perform a continuous greedy algorithm to maximize the objective function.

    Args:
        k (int, optional): The number of top elements to select. Defaults to 10.

    Returns:
        numpy.ndarray: Indices of the top-k elements in B_to_A according to the greedy algorithm.
    """
    # Initialize vect_x as a zero vector of the same length as B_to_A.
    vect_x = np.zeros(len(B_to_A))

    # Iteratively update vect_x using the gradient, for k iterations.
    for _ in range(k):
        vect_v = dist_sample_gradient(A_to_B, B_to_A, vect_x)
        vect_x += 1.0 / k * vect_v

    # Return the indices of the top-k elements in vect_x.
    return np.argsort(-vect_x)[:k]
