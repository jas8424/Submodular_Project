import numpy as np
import random


def dist_sample_cont_greedy(A_to_B, B_to_A, k=20, p=50, sample_rate=0.2):
    # Initializing
    A_length, B_length = len(A_to_B), len(B_to_A)
    sampled_A = int(A_length * sample_rate)
    vect_x = np.zeros(B_length)
    A_to_B_items = list(A_to_B.items())

    for _ in range(k):
        gradient_matrix = np.zeros((sampled_A, B_length))
        matrix_v = np.zeros((sampled_A, B_length), dtype=np.bool_)
        # Sampled professors
        sampled_AB_items = random.sample(A_to_B_items, sampled_A)

        # Estimate the gradient, assume it's correct, explain later
        for _ in range(p):
            sampled_Bs = np.nonzero(np.random.rand(B_length) <= vect_x)[0]

            for i, (key, A_to_B_value) in enumerate(sampled_AB_items):
                intersect_index = np.intersect1d(A_to_B_value, sampled_Bs, assume_unique=True)
                if not intersect_index.size:
                    gradient_matrix[i, np.setdiff1d(A_to_B_value, sampled_Bs, assume_unique=True)] += 1 / p
                elif len(intersect_index) == 1:
                    gradient_matrix[i, intersect_index] += 1 / p

        # Update vector_v
        top_k_indices = np.argsort(-gradient_matrix, axis=1)[:, :k]
        row_indices = np.arange(sampled_A)[:, None]
        matrix_v[row_indices, top_k_indices] = 1
        vect_v = np.mean(matrix_v, axis=0)

        vect_x += 1.0 / k * vect_v
    return vect_x