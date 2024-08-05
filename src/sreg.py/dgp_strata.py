import numpy as np

def form_strata_sreg(baseline, num_strata):
    n = len(baseline['Y_0'])
    W = baseline['W']
    bounds = np.linspace(-2.25, 2.25, num_strata + 1)
    I_S = np.zeros((n, num_strata))

    for s in range(num_strata):
        I_S[:, s] = (W > bounds[s]) & (W <= bounds[s + 1])

    return I_S

def form_strata_creg(baseline, num_strata):
    n = baseline['G']
    W = baseline['Z_g_2']
    bounds = np.linspace(min(W), max(W), num_strata + 1)
    I_S = np.zeros((n, num_strata))

    for s in range(num_strata):
        I_S[:, s] = (W > bounds[s]) & (W <= bounds[s + 1])

    return I_S
