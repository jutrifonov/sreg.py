import numpy as np
import pandas as pd

def dgp_po_sreg(n, theta_vec, gamma_vec, n_treat, is_cov=True):
    if n_treat != len(theta_vec):
        raise ValueError("The number of treatments doesn't match the length of vector theta_vec.")

    mu = {f"mu_{a}": theta_vec[a-1] for a in range(1, n_treat + 1)}
    
    eps_0 = np.random.normal(size=n)
    eps = {f"eps_{a}": np.random.normal(size=n) for a in range(1, n_treat + 1)}

    W = np.sqrt(20) * (np.random.beta(2, 2, size=n) - 0.5)
    x_1 = np.random.normal(5, 2, size=n)
    x_2 = np.random.normal(2, 1, size=n)

    if is_cov:
        X = pd.DataFrame({'x_1': x_1, 'x_2': x_2})
        m_0 = gamma_vec[0] * W + gamma_vec[1] * x_1 + gamma_vec[2] * x_2
    else:
        X = None
        m_0 = gamma_vec[0] * W

    m = {f"m_{a}": m_0 for a in range(n_treat + 1)}

    Y_0 = m_0 + eps_0
    Y = {f"Y_{a}": mu[f"mu_{a}"] + m[f"m_{a}"] + eps[f"eps_{a}"] for a in range(1, n_treat + 1)}
    Y['Y_0'] = Y_0

    if is_cov:
        ret_names = [f"Y_{a}" for a in range(n_treat + 1)] + ['W', 'X'] + [f"m_{a}" for a in range(n_treat + 1)] + [f"mu_{a}" for a in range(1, n_treat + 1)]
        ret_list = {**Y, 'W': W, 'X': X, **m, **mu}
    else:
        ret_names = [f"Y_{a}" for a in range(n_treat + 1)] + ['W'] + [f"m_{a}" for a in range(n_treat + 1)] + [f"mu_{a}" for a in range(1, n_treat + 1)]
        ret_list = {**Y, 'W': W, **m, **mu}

    return ret_list


# n = 100
# theta_vec = [0.5, 1.5]
# gamma_vec = [0.1, 0.2, 0.3]
# n_treat = 2
# is_cov = True

# np.random.seed(123)
# result = dgp_po_sreg(n, theta_vec, gamma_vec, n_treat, is_cov)
# result['Y_1']


def dgp_po_creg(Ng, G, tau_vec, sigma1=np.sqrt(2), gamma_vec=[0.4, 0.2, 1], n_treat=None):
    if n_treat is None:
        n_treat = len(tau_vec)
    
    if n_treat != len(tau_vec):
        raise ValueError("The number of treatments doesn't match the length of vector tau_vec.")
    
    mu_dict = {f"mu_{a+1}": tau for a, tau in enumerate(tau_vec)}

    beta_rv = np.random.beta(2, 2, G)
    Z_g_2 = (beta_rv - 0.5) * np.sqrt(20)
    x_1 = (np.random.normal(5, 2, G) - 5) / 2
    x_2 = (np.random.normal(2, 1, G) - 2) / 1
    X = pd.DataFrame({'x_1': x_1, 'x_2': x_2})

    cluster_indicator = np.repeat(np.arange(1, G+1), Ng)
    cl_id = cluster_indicator
    total_sample = len(cluster_indicator)

    epsilon_ig_dict = {f"epsilon_ig_{a+1}": np.random.normal(0, sigma1, total_sample) for a in range(n_treat)}
    epsilon_ig_0 = np.random.normal(0, 1, total_sample)

    m_0 = gamma_vec[0] * Z_g_2 + gamma_vec[1] * x_1 + gamma_vec[2] * x_2
    m_0_repeated = np.repeat(m_0, Ng)

    m_dict = {f"m_{a+1}": m_0 for a in range(n_treat)}

    Yig_0 = m_0_repeated + epsilon_ig_0

    Yig_dict = {}
    for a in range(1, n_treat+1):
        formula = mu_dict[f"mu_{a}"] + m_0_repeated + epsilon_ig_dict[f"epsilon_ig_{a}"]
        Yig_dict[f"Yig_{a}"] = formula

    ret_dict = {**Yig_dict, 'Yig_0': Yig_0, 'Z_g_2': Z_g_2, 'X': X, 'G': G, 'Ng': Ng, 'cl_id': cl_id, **m_dict, **mu_dict}

    return ret_dict

# # Example usage
# n = 5
# Ng = np.random.randint(1, 4, size=3)  # Ensure Ng length matches G
# G = 3
# tau_vec = [0.5, 1.5]
# gamma_vec = [0.4, 0.2, 1]
# n_treat = 2

# np.random.seed(0)
# result = dgp_po_creg(Ng, G, tau_vec, gamma_vec=gamma_vec, n_treat=n_treat)
# print(result)
# result['Yig_0']