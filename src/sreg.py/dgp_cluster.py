import numpy as np

def gen_cluster_sizes(G, max_support):
    alpha = 1
    beta = 1
    sample = 10 * (np.random.beta(alpha, beta, G) * max_support).astype(int) + 10
    return sample

# # Example usage\
# Nmax = 50
# max_support = Nmax / 10 - 1
# G = 100
# np.random.seed(0)
# result = gen_cluster_sizes(G, max_support)
# print(result)