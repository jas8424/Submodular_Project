import random
import heapq

def coverage(B_to_A, venues_set):
    """
    Calculates the coverage of a set of venues.

    Parameters:
    - B_to_A (dict): A dictionary mapping each venue to a set of entities (e.g., users or benefits) it covers.
    - venues_set (set): A set of venues for which to calculate the total coverage.

    Returns:
    - int: The total number of unique entities covered by the given set of venues.
    """
    if not venues_set:
        return 0  # Return 0 if the input set is empty

    # Unite all sets of entities covered by each venue in the venues_set
    united_set = set.union(*[B_to_A[i] for i in venues_set])

    return len(united_set)  # Return the size of the united set, representing total coverage



def gradient(B_to_A, vect_x, index_x, p=25):
    """
    Computes the gradient of the coverage function with respect to each venue.

    Parameters:
    - B_to_A (dict): Mapping of venues to the entities they cover.
    - vect_x (list): Probability vector, where each element corresponds to the likelihood of including a venue.
    - index_x (set): Indices of venues with non-zero probabilities in vect_x.
    - p (int): Number of iterations for estimating the gradient via sampling.

    Returns:
    - list: A list representing the gradient of the coverage function with respect to each venue.
    """
    length = len(B_to_A)
    gradient_list = [0] * length

    for _ in range(p):
        # Sample venues based on their probabilities in vect_x
        sampled_venues = set(i for i in index_x if random.random() <= vect_x[i])
        temp = coverage(B_to_A, sampled_venues)  # Compute the base coverage

        for i in range(length):
            # Evaluate the marginal contribution of including or excluding venue i
            with_i = sampled_venues | {i}
            if i in sampled_venues:
                without_i = sampled_venues - {i}
                difference = temp - coverage(B_to_A, without_i)
            else:
                difference = coverage(B_to_A, with_i) - temp
            gradient_list[i] += difference / p  # Average the marginal contributions

    return gradient_list



def cont_greedy(B_to_A, k=10):
    """
    Performs the continuous greedy algorithm to select a set of venues maximizing coverage.

    Parameters:
    - B_to_A (dict): Mapping from venues to the entities they cover.
    - k (int): The number of venues to select for maximizing coverage.

    Returns:
    - list: Indices of the top `k` venues selected by the algorithm.
    """
    vect_x = [0] * len(B_to_A)  # Initialize the selection probabilities of venues to 0
    index_x = set()  # Initialize an empty set to track indices with non-zero probabilities

    for _ in range(k):
        vect_v = gradient(B_to_A, vect_x, index_x)  # Compute the gradient of the coverage
        # Identify the top k venues based on their marginal contribution to the coverage
        top_indices = heapq.nlargest(k, range(len(vect_v)), key=lambda i: vect_v[i])

        for j in top_indices:
            # Update the probability vector and the set of active indices
            if vect_x[j] == 0:
                index_x.add(j)
            vect_x[j] += 1 / k

    # Return the indices of the top k venues based on their final selection probabilities
    return heapq.nlargest(k, range(len(vect_x)), key=lambda i: vect_x[i])
