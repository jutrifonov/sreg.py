# sreg_script.py
import pandas as pd
import numpy as np
from sreg import sreg, sreg_rgen  # Ensure these imports work in your Python environment

from sreg import sreg_rgen

def run_sreg_rgen(n, Nmax, n_strata, tau_vec, gamma_vec, cluster, is_cov):
    return sreg_rgen(n=n, Nmax=Nmax, n_strata=n_strata, tau_vec=tau_vec, gamma_vec=gamma_vec, cluster=cluster, is_cov=is_cov)