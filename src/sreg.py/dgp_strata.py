import numpy as np

def form_strata_sreg(baseline, num_strata):
    n = len(baseline['Y_0'])
    W = baseline['W']
    bounds = np.linspace(-2.25, 2.25, num_strata + 1)
    I_S = np.zeros((n, num_strata))

    for s in range(num_strata):
        I_S[:, s] = (W > bounds[s]) & (W <= bounds[s + 1])

    return I_S

# Example usage
# n = 100
# theta_vec = [0.5, 1.5]
# gamma_vec = [0.1, 0.2, 0.3]
# n_treat = 2
# is_cov = True

# np.random.seed(123)
# baseline = dgp_po_sreg(n, theta_vec, gamma_vec, n_treat, is_cov)
# num_strata = 5
# strata = form_strata_sreg(baseline, num_strata)
# strata_set = pd.DataFrame(strata)
# strata_set['S'] = strata_set.idxmax(axis=1) + 1  # Add 1 to match R's 1-based indexing

# print(strata_set)

def form_strata_creg(baseline, num_strata):
    n = baseline['G']
    W = baseline['Z_g_2']
    bounds = np.linspace(min(W), max(W), num_strata + 1)
    I_S = np.zeros((n, num_strata))

    for s in range(num_strata):
        I_S[:, s] = (W > bounds[s]) & (W <= bounds[s + 1])

    return I_S

# # # Example usage
# n = 100
# G = n
# Nmax = 50
# max_support = Nmax / 10 - 1
# Ng = gen_cluster_sizes(G, max_support)

# tau_vec = [0.5, 1.5]
# gamma_vec = [0.4, 0.2, 1]
# n_treat = 2

# np.random.seed(0)
# baseline = dgp_po_creg(Ng, G, tau_vec, gamma_vec=gamma_vec, n_treat=n_treat)
# num_strata = 5
# strata = form_strata_creg(baseline, num_strata)

# # Convert to DataFrame and add column S with max col index
# strata_set = pd.DataFrame(strata)
# strata_set['S'] = strata_set.idxmax(axis=1) + 1  # Add 1 to match R's 1-based indexing

# print(strata_set)