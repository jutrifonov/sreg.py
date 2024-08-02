import numpy as np

# Example usage
n = 100
theta_vec = [0.5, 1.5, 2]
gamma_vec = [0.1, 0.2, 0.3]
n_treat = 3
is_cov = True


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

def gen_treat_sreg(pi_matr_w, ns, k):
    rows = pi_matr_w.shape[0]
    code_elements = []

    for i in range(rows):
        code_elements.append(
            np.full(int(np.floor(pi_matr_w[i, k - 1] * ns)), i + 1)
        )

    remaining_count = ns - int(sum(np.floor(pi_matr_w[i, k - 1] * ns) for i in range(rows)))
    code_elements.append(np.full(remaining_count, 0))

    # Concatenate the arrays
    result = np.concatenate(code_elements)
    np.random.shuffle(result)

    return result

# Main loop to assign treatments
for k in range(1, num_strata + 1):
    index = np.where(I_S[:, k - 1] == 1)[0]
    ns = len(index)

    A[index] = gen_treat_sreg(pi_matr_w, ns, k)

# Example usage
print(A)

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


def gen_treat_creg(pi_matr_w, ns, k):
    rows = pi_matr_w.shape[0]
    code_elements = []

    for i in range(rows):
        code_elements.append(
            np.full(int(np.floor(pi_matr_w[i, k - 1] * ns)), i + 1)
        )

    remaining_count = ns - int(sum(np.floor(pi_matr_w[i, k - 1] * ns) for i in range(rows)))
    code_elements.append(np.full(remaining_count, 0))

    # Concatenate the arrays
    result = np.concatenate(code_elements)
    np.random.shuffle(result)

    return result

# Main loop to assign treatments
for k in range(1, num_strata + 1):
    index = np.where(I_S[:, k - 1] == 1)[0]
    ns = len(index)

    A[index] = gen_treat_sreg(pi_matr_w, ns, k)

# Example usage
print(A)
