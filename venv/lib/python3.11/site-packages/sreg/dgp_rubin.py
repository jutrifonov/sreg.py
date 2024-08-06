def gen_rubin_formula_sreg(n_treat):
    A_values = range(n_treat + 1)

    formula = "Y_obs = "

    for a in A_values:
        if a == 0:
            formula += f"Y_{a} * (A == 0)"
        else:
            formula += f"Y_{a} * (A == {a})"

        if a < n_treat:
            formula += " + "

    return formula

def gen_rubin_formula_creg(n_treat):
    A_values = range(n_treat + 1)

    formula = "Y_obs = "

    for a in A_values:
        if a == 0:
            formula += f"Y_{a} * (A == 0)"
        else:
            formula += f"Y_{a} * (A == {a})"

        if a < n_treat:
            formula += " + "

    return formula
