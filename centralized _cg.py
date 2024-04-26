def centralized _cg(B_to_A, k=10):
    """
    Try to find the maximum coverage of nodes in set A by selecting nodes from set B.

    Args:
        B_to_A (dict): Venues(int) are keys and Professors(set()) who published on that venues are corresponding values
        k (int, optional): The maximum number of nodes to select from set B. Default is 10.

    Returns:
        - coverage_record (list): A list of cumulative counts of covered nodes.
    """
    covered = set()  # Set to store the covered nodes in set A
    selected_B = set()  # Set to store the selected nodes from set B
    coverage_gain = {key: len(value) for key, value in B_to_A.items()}  # Dictionary to store the potential coverage gain of each node in set B
    coverage_record = []  # List to store the cumulative counts of covered nodes from 1 to k

    for i in range(k):
        if not coverage_gain:  # If there are no more nodes to select
            break

        best_node = max(coverage_gain, key=coverage_gain.get)  # Find the node with the maximum potential coverage gain

        if coverage_gain[best_node] == 0:  # If the potential coverage gain is 0, no additional coverage can be achieved
            break

        selected_B.add(best_node)  # Add the best node to the selected set
        covered.update(B_to_A[best_node])  # Add the covered nodes to the covered set
        coverage_record.append(len(covered))  # Record the cumulative number of covered nodes

        for node in coverage_gain:
            if node not in selected_B:
                coverage_gain[node] = len(B_to_A[node] - covered)  # Update the potential coverage gain for the remaining nodes

        del coverage_gain[best_node]  # Remove the selected node from the coverage gain dictionary

    return coverage_record
