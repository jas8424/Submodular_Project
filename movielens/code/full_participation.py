import numpy as np
import time
from scipy.optimize import linprog
import random
import pdb
# seed=2
# random.seed(seed)
# np.random.seed(seed)
rng = np.random.default_rng()

R = np.load('/Users/akbarrafiey/Documents/fl-submodular/movielens-24/data/ratings.npy')


def full_participation(R, num_iteration, cardinality, num_gradient_estimation_samples):
    '''
    Performs a dicentralized continuous greedy algorithm with full participation of clients to select a subset of movies based on ratings.
    
    This function simulates a greedy maximization process constrained by cardinality and uses multiple
    samples to estimate gradients, adapting ideas from submodular function maximization in a continuous setting.
    
    Parameters:
    - R (np.ndarray): A 2D numpy array where each row corresponds to a user and each column to a movie rating.
    - num_iterations (int): The number of iterations to run the  continuous greedy algorithm.
    - cardinality (int): The maximum number of movies to be selected.
    - num_gradient_estimation_samples (int): The number of random samples to use for estimating the gradient.
    
    Returns:
    - function value
    '''
    
    # # Number of random rows and columns to select
    # num_rows = 5
    # num_cols = 10

    # # Generate random indices
    # row_indices = np.random.choice(R.shape[0], num_rows, replace=False)
    # col_indices = np.random.choice(R.shape[1], num_cols, replace=False)

    # # Select the sub-matrix
    # R = R[row_indices[:, None], col_indices]
    
    eta = 1 / num_iteration
    num_clients = R.shape[0]
    num_movies = R.shape[1]
    # R_nz_index = np.argwhere(R)
    # num_nonzero = R_nz_index.shape[0]
    
    x = np.zeros(num_movies, dtype=np.float64)

    for t in range(num_iteration):
        start = time.time()
        # df = np.zeros((num_clients, num_movies))

        # sample random sets according to x
        # number of random sets = num_gradient_estimation_samples
        random_sets = rng.binomial(n=1, p=x, size=(num_gradient_estimation_samples, num_movies)) == 1
        
        delta = np.zeros(num_movies, dtype=np.float64)
        for client in range(num_clients):
        # estimating the gradient vector
            grad_client = np.zeros(num_movies, dtype=np.float64)
    
            
            for movie in range(num_movies):
                for random_set in random_sets :
                    # pdb.set_trace()        

                    S = random_set
                    
                    # compute F(S + movie)
                    S[movie] = 1
                    a = np.amax(R[client, S])
                    
                    # compute F(S - movie)
                    S[movie] = 0
                    if not np.any(S): # check if the set is empty
                        b = 0
                    else:
                        b = np.amax(R[client, S])
                    
                    #b = np.sum(np.amax(R[:, random_set], axis=1))         
                    grad_client[movie] += a - b
                    
            
            # Find the indices of the top k entries
            top_k_indices = np.argsort(grad_client)[-cardinality:]  # Get the indices of the top k values

            v_client = np.zeros_like(grad_client, dtype=int)

            # Set the top k indices to 1
            v_client[top_k_indices] = 1

            # update delta
            delta += v_client

        
        # scale delta
        delta = delta / num_clients
        # update x
        x = x + eta * delta
        

        print(t+1, 'th iteration is done')
        end = time.time()
        print('The running time is', end - start)


    output_set = greedy_rounding(x, cardinality)
    function_value = np.sum(np.amax(R[:, output_set], axis=1)) / num_clients
    
    return function_value

def greedy_rounding(x, k):
    """
    Returns a 0/1 vector v of the same length as x, where v has exactly k ones
    at the indices of the top k values of x.

    Parameters:
    x (np.array): A numpy array of numeric values.
    k (int): The number of top entries to set to 1 in the resulting vector.

    Returns:
    np.array: A Boolean vector where the top k entries of x are set to 1.
    """
    # Ensure k is not greater than the length of x
    if k > len(x):
        raise ValueError("k cannot be greater than the number of elements in x")

    # Initialize the vector v with zeros
    v = np.zeros_like(x, dtype=bool)

    # Find the indices of the top k values in x
    top_k_indices = np.argsort(x)[-k:]

    # Set the indices of the top k values to 1 in vector v
    v[top_k_indices] = True

    return v


def f_value(R, x):

    R_sparse = R[:, x]
    f = np.sum(np.amax(R_sparse, axis=1))

    return f

vector_k = np.array([3, 5, 10, 15, 20, 25])
f_k = np.zeros((len(vector_k)))

for i in range(len(vector_k)):
    k = vector_k [i]
    value = full_participation(R, num_iteration=10, cardinality = k, num_gradient_estimation_samples = 2)
    f_k[i] = value
    print(i+1, 'th function value is obtained: ', value)

print(f_k)
#np.save(str(seed)+"f_centralized.npy", f_k)
