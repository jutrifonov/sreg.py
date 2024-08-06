import numpy as np

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

