# %#     by taking as input the potential outcomes,
# %#     matrix of strata assignments, pi.vec, and
# %#     number of treatments
import numpy as np
import pandas as pd


def dgp_obs_sreg(baseline, I_S, pi_vec, n_treat, is_cov=True):
    if n_treat != len(pi_vec):
        raise ValueError("The number of treatments doesn't match the length of vector pi.vec.")

    num_strata = I_S.shape[1]
    n = len(baseline['Y_0'])
    A = np.zeros(n, dtype=int)
    l_seq = num_strata / 2

    pi_matr = np.ones((n_treat, num_strata))
    pi_matr_w = pi_matr * pi_vec

    for k in range(num_strata):
        index = np.where(I_S[:, k] == 1)[0]
        ns = len(index)
        A[index] = gen_treat_sreg(pi_matr_w, ns, k)

    Y_obs = np.zeros(n)
    for a in range(n_treat + 1):
        Y_obs += baseline[f"Y_{a}"] * (A == a)

    if is_cov:
        ret_list = {
            "Y": Y_obs,
            "D": A,
            "X": baseline['X']
        }
    else:
        ret_list = {
            "Y": Y_obs,
            "D": A
        }
    return ret_list



baseline = dgp_po_sreg(n, theta_vec, gamma_vec, n_treat, is_cov)
# Initial parameters

pi_vec = np.full(n_treat, 1 / (n_treat + 1))
num_strata=3
I_S = form_strata_sreg(baseline, num_strata)
# Assuming I_S and baseline are already defined
num_strata = I_S.shape[1]
n = len(baseline['Y_0'])

A = np.zeros(n, dtype=int)
l_seq = num_strata / 2

pi_matr = np.ones((n_treat, num_strata))
pi_matr_w = pi_matr * pi_vec[:, np.newaxis]

# Main loop to assign treatments
for k in range(1, num_strata + 1):
    index = np.where(I_S[:, k - 1] == 1)[0]
    ns = len(index)

    A[index] = gen_treat_sreg(pi_matr_w, ns, k)

# Example usage
print(A)

obs_otcm=dgp_obs_sreg(baseline, I_S, pi_vec, n_treat, is_cov=True)
obs_otcm['D']

def dgp_obs_creg(baseline, I_S, pi_vec, n_treat):
    if n_treat != len(pi_vec):
        raise ValueError("The number of treatments doesn't match the length of vector pi.vec.")

    num_strata = I_S.shape[1]
    n = baseline['G']
    A = np.zeros(n, dtype=int)
    l_seq = num_strata / 2

    pi_matr = np.ones((n_treat, num_strata))
    pi_vec = np.array(pi_vec).reshape(-1, 1)  # Reshape pi_vec for broadcasting
    pi_matr_w = pi_matr * pi_vec

    for k in range(num_strata):
        index = np.where(I_S[:, k] == 1)[0]
        ns = len(index)
        A[index] = gen_treat_creg(pi_matr_w, ns, k)

    strata_set = pd.DataFrame(I_S)
    strata_set['S'] = strata_set.idxmax(axis=1) + 1
    cluster_indicator = baseline['cl_id']
    G_seq = np.arange(1, baseline['G'] + 1)
    data_short = pd.DataFrame({
        "cl_id": G_seq, 
        "A": A, 
        "S": strata_set['S'], 
        "Ng": baseline['Ng']
    }).join(baseline['X'])

    data_long = pd.DataFrame({"cl_id": cluster_indicator})
    merged_data = data_long.merge(data_short, on="cl_id")
    A = merged_data['A'].values
    S = merged_data['S'].values
    X = merged_data.iloc[:, 5:].values
    Ng = merged_data['Ng'].values

    Y_obs = np.zeros(len(A))
    for a in range(n_treat + 1):
        Y_obs += baseline[f"Yig_{a}"] * (A == a)

    ret_list = {
        "Y": Y_obs,
        "D": A,
        "S": S,
        "Z_2": baseline['Z_g_2'],
        "X": X,
        "Ng": Ng,
        "G_id": cluster_indicator,
        "cl_lvl_data": data_short
    }
    
    return ret_list

# Example usage
np.random.seed(42)  # For reproducibility



n = 100
G = n
Nmax = 50
max_support = Nmax / 10 - 1
Ng = gen_cluster_sizes(G, max_support)

tau_vec = [0.5, 1.5]
gamma_vec = [0.4, 0.2, 1]
n_treat = 2

np.random.seed(0)
baseline = dgp_po_creg(Ng, G, tau_vec, gamma_vec=gamma_vec, n_treat=n_treat)

# Initial parameters

pi_vec = np.full(n_treat, 1 / (n_treat + 1))
num_strata=3
I_S = form_strata_creg(baseline, num_strata)
# Assuming I_S and baseline are already defined
num_strata = I_S.shape[1]
n = len(baseline['Yig_0'])

A = np.zeros(n, dtype=int)
l_seq = num_strata / 2

pi_matr = np.ones((n_treat, num_strata))
pi_matr_w = pi_matr * pi_vec[:, np.newaxis]

# Main loop to assign treatments
for k in range(1, num_strata + 1):
    index = np.where(I_S[:, k - 1] == 1)[0]
    ns = len(index)

    A[index] = gen_treat_sreg(pi_matr_w, ns, k)

# Example usage
print(A)

obs_creg=dgp_obs_creg(baseline, I_S, pi_vec, n_treat)
obs_creg