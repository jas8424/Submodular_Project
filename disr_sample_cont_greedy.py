import numpy as np
import random

# def dist_sample_cont_greedy(A_to_B, B_to_A, communication rounds ,k=5, =50, sample_rate=0.2):
# k is the number of conferences we select at the end
# each vector v can have at most k ones in it
# p should be named num_gradient_estimation_samples 
# sample_rate is number of clieatns or professors we sample at each round--> better name it client_sample


def dist_sample_cont_greedy(A_to_B, B_to_A, t=40, k=20, num_gradient_estimation_samples=20, client_sample = 0.1):
    A_length, B_length = len(A_to_B), len(B_to_A)
    sampled_A = int(A_length * client_sample)
    vect_x = np.zeros(B_length)
    A_to_B_items = list(A_to_B.items())

    for _ in range(t):
        gradient_matrix = np.zeros((sampled_A, B_length), dtype=np.int8)
        matrix_v = np.zeros((sampled_A, B_length), dtype=np.bool_)
        sampled_AB_items = random.sample(A_to_B_items, sampled_A)
        
        for _ in range(num_gradient_estimation_samples):
            sampled_Bs = np.nonzero(np.random.rand(B_length) <= vect_x)[0]
            for i, (key, A_to_B_value) in enumerate(sampled_AB_items):
                intersect_index = np.intersect1d(A_to_B_value, sampled_Bs, assume_unique=True)
                if not intersect_index.size:
                    gradient_matrix[i, np.setdiff1d(A_to_B_value, sampled_Bs, assume_unique=True)] += 1
                elif len(intersect_index) == 1:
                    gradient_matrix[i, intersect_index] += 1

        # Apply random shuffle to indices
        shuffled_indices = np.random.permutation(B_length)

        # Apply argpartition to get top k indices in shuffled order
        partitioned_indices = np.argpartition(-gradient_matrix[:, shuffled_indices], kth=k-1, axis=1)[:, :k]

        # Convert shuffled partitioned indices back to original indices
        original_top_k_indices = shuffled_indices[partitioned_indices]

        row_indices = np.arange(sampled_A)[:, None]
        matrix_v[row_indices, original_top_k_indices] = 1
        vect_v = np.mean(matrix_v, axis=0)

        vect_x += 1.0 / t * vect_v
        
    return vect_x
