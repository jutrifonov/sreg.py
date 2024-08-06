import numpy as np

def gen_cluster_sizes(G, max_support):
    alpha = 1
    beta = 1
    sample = 10 * (np.random.beta(alpha, beta, G) * max_support).astype(int) + 10
    return sample
