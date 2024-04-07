import numpy as np
import random
import heapq

def dist_gradient(A_to_B, B_to_A, vect_x, index_x, p=1):
    """
    Computes the gradient of the objective function with optimizations, useful for
    gradient-based optimization algorithms in distributed systems or networked applications.

    Parameters:
    - A_to_B (dict): Mapping from entities in A to sets of entities in set B, representing connections or relations.
    - B_to_A (dict): Mapping from entities in B to sets of entities in set A
    - vect_x (list): Current state vector in the optimization process, indicating the selection probabilities.
    - index_x (set): Indices in vect_x that are currently active (non-zero).
    - p (int): Number of iterations for averaging the gradient.

    Returns:
    - np.array: Averaged gradient vector over `p` iterations, for use in optimization updates.
    """
    # Initialize the gradient matrix with zeros
    length = len(A_to_B)
    gradient_list = np.zeros((length, len(B_to_A)))
    learning_rate = 1 / p

    for _ in range(p):
        # Sample venues based on current probabilities in vect_x
        sampled_venues = {i for i in index_x if random.random() <= vect_x[i]}

        for i in range(length):
            # Pre-compute intersections to avoid redundant calculations
            A_to_B_i = A_to_B[i]
            intersection_with_sample = A_to_B_i.intersection(sampled_venues)
            for k in B_to_A:
                # Increment gradient for relevant combinations of A and B entities
                if not intersection_with_sample and k in A_to_B_i:
                    gradient_list[i, k] += learning_rate
                elif k in A_to_B_i and k not in sampled_venues:
                    gradient_list[i, k] += learning_rate

    # Return the mean gradient across all entities in A
    return np.mean(gradient_list, axis=0)

import heapq

def dist_cont_greedy(A_to_B, B_to_A, k=10):
    """
    Performs the distributed continuous greedy algorithm to optimize selections
    based on gradients computed from relations between two sets A and B.

    Parameters:
    - A_to_B (dict): Mapping from entities in A to sets of entities in set B.
    - B_to_A (dict): Mapping from entities in B to sets of entities in set A.
    - k (int): Specifies the number of top elements to select based on the optimization process.

    Returns:
    - list: Indices of the top `k` elements in the final state vector, indicating the optimized selections.
    """
    # Initialize the state vector with zeros and an empty set for tracking active indices
    vect_x = [0] * len(B_to_A)
    index_x = set()

    for _ in range(k):
        # Compute the gradient for current state
        vect_v = dist_gradient(A_to_B, B_to_A, vect_x, index_x)
        # Find the top k indices based on the computed gradient
        top_indices = heapq.nlargest(k, range(len(vect_v)), key=lambda i: vect_v[i])

        for j in top_indices:
            # Update active index set and incrementally adjust state vector based on the gradient
            if vect_x[j] == 0:
                index_x.add(j)
            vect_x[j] += 1 / k * vect_v[j]

    # Return indices of the top k elements after optimization
    return heapq.nlargest(k, range(len(vect_x)), key=lambda i: vect_x[i])
