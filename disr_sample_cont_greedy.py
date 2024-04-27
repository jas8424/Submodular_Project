import numpy as np
import random

def dist_sample_cont_greedy(A_to_B, B_length, t=40, k=20, num_gradient_estimation_samples=10, client_sample=0.1):
    """
    This function implements a distributed sampling continuous greedy algorithm.

    Args:
        A_to_B (dict): A dictionary where keys are A(professors), and values are list of B(venues) that A have.
        B_length (int):  Length of B (venues)
        t (int): Number of communication rounds.
        k (int): Number of top elements to keep for vector v.
        num_gradient_estimation_samples (int): Number of samples to use for gradient estimation.
        client_sample (float): Fraction of elements from set A to sample.

    Returns:
        np.ndarray: A vector of length len(B_to_A) representing the matching scores for elements in set B.
    """
    A_length = len(A_to_B)
    sampled_A = int(A_length * client_sample)
    vect_x = np.zeros(B_length) 
    A_to_B_values = list(A_to_B.values()) 
    for _ in range(t):
        gradient_matrix = np.zeros((sampled_A, B_length), dtype=np.int16)  # Initialize gradient matrix
        matrix_v = np.zeros((sampled_A, B_length), dtype=np.bool_)  # Initialize boolean matrix

        # Sample elements from A_to_B if client_sample is not 1, else use all elements
        sampled_AB_values = random.sample(A_to_B_values, sampled_A) if client_sample != 1 else A_to_B_values

        for _ in range(num_gradient_estimation_samples):
            sampled_Bs = np.flatnonzero(np.random.rand(B_length) <= vect_x)  # Sample B(venues)base on the distribution of vect_x
            for i, A_to_B_value in enumerate(sampled_AB_values):
                intersect_index = np.intersect1d(A_to_B_value, sampled_Bs, assume_unique=True)
                if not intersect_index.size:
                    # If no intersection, increment gradients for values in A_to_B_value
                    gradient_matrix[i, np.setdiff1d(A_to_B_value, sampled_Bs, assume_unique=True)] += 1
                elif intersect_index.size == 1:
                    # If one intersection, increment gradients for intersected and other values in A_to_B_value
                    gradient_matrix[i, intersect_index] += 1
                    gradient_matrix[i, np.setdiff1d(A_to_B_value, sampled_Bs, assume_unique=True)] += 1

        # Select top k elements based on gradient for each element in sampled_AB_values since that's the only entries that not 0
        for i, A_to_B_value in enumerate(sampled_AB_values):
            top_k_indices = A_to_B_value[np.argsort(-gradient_matrix[i, A_to_B_value])[:k]]
            matrix_v[i, top_k_indices] = True

        vect_v = np.mean(matrix_v, axis=0)  # Calculate the mean of matrix_v along the first axis
        vect_x += vect_v / t  # Update the matching score vector

    return vect_x
