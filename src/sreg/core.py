"""
Sreg:Stratified Randomized Experiments
============

This package provides tools for performing XYZ operations. It includes modules for A, B, and C.
"""
# The core
import pandas as pd
import numpy as np

from .data_check import check_data_types, check_integers, boolean_check, check_range
from .check_cluster import check_cluster, check_cluster_lvl
from .result import res_sreg, res_creg
from .dgp_cluster import gen_cluster_sizes
from .dgp_po import dgp_po_sreg, dgp_po_creg
from .dgp_strata import form_strata_sreg, form_strata_creg
from .dgp_obs import dgp_obs_sreg, dgp_obs_creg



def sreg(Y, S=None, D=None, G_id=None, Ng=None, X=None, HC1=True):
    """Sreg provides estimates of ATE, corresponding st.errors and CI."""
    check_data_types(Y, S, D, G_id, Ng, X)
    check_integers(S, D, G_id, Ng)
    boolean_check(HC1)

    if Y is None:
        raise ValueError("Error: Observed outcomes have not been provided (Y = None). Please provide the vector of observed outcomes.")
    if D is None:
        raise ValueError("Error: Treatments have not been provided (D = None). Please provide the vector of treatments.")
    if D is not None:
        check_range(D)
    if S is not None:
        check_range(S)

    if X is not None and 'Ng' in X.columns:
        X = X.rename(columns={'Ng': 'Ng_1'})
    if X is not None:
        if isinstance(X, pd.DataFrame):
            X_dict = {col: X[col].values for col in X.columns}
        elif isinstance(X, np.ndarray):
            X_dict = {f'X{i}': X[:, i] for i in range(X.shape[1])}
        else:
            raise ValueError("X should be a pandas DataFrame or a NumPy array.")
    else:
        X_dict = {}
    #if Ng is not None:
    check_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng, **X_dict})
    if G_id is None: 
        check_df=check_df.drop(columns=['G_id'])

    if S is None:
        check_df=check_df.drop(columns=['S'])
    if Ng is None:
        check_df=check_df.drop(columns=['Ng'])

    if check_df.isnull().any().any():
        print("Warning: The data contains one or more NA (or NaN) values. Proceeding while ignoring these values.")
    clean_df = check_df.dropna()

    x_ind = max([i for i, col in enumerate(clean_df.columns) if col in ['D', 'G_id', 'Ng']], default=-1)

    Y = clean_df['Y']
    if S is not None:
        S = clean_df['S']
    D = clean_df['D']
    if G_id is not None:
        G_id = clean_df['G_id']
    if Ng is not None:
        Ng = clean_df['Ng']
    
    if (x_ind + 1) >= len(clean_df.columns):
        X = None
    else:
        X = clean_df.iloc[:, (x_ind + 1):]

    if X is not None and 'Ng_1' in X.columns:
        X = X.rename(columns={'Ng_1': 'Ng'})

    if S is not None:
        if S.min() != 1:
            raise ValueError(f"Error: The strata should be indexed by {{1, 2, 3, ...}}. The minimum value in the provided data is {S.min()}.")
    if D is not None:
        if D.min() != 0:
            raise ValueError(f"Error: The treatments should be indexed by {{0, 1, 2, ...}}, where D = 0 denotes the control. The minimum value in the provided data is {D.min()}.")

    if G_id is None:
        result = res_sreg(Y, S, D, X, HC1)
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
    else:
        check_cluster_lvl(G_id, S, D, Ng)
        result = res_creg(Y, S, D, G_id, Ng, X, HC1)
        if Ng is None:
            print("Warning: Cluster sizes have not been provided (Ng = None). Ng is assumed to be equal to the number of available observations in every cluster g.")
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
        if result['lin_adj'] is not None:
            if not check_cluster(pd.DataFrame({'G_id': result['data']['G_id'], **result['lin_adj']})):
                print("Warning: sreg cannot use individual-level covariates for covariate adjustment in cluster-randomized experiments. Any individual-level covariates have been aggregated to their cluster-level averages.")

    return result

def sreg_rgen(n, Nmax=50, n_strata=5, tau_vec=[0], gamma_vec=[0.4, 0.2, 1], cluster=True, is_cov=True):
    if cluster:
        G = n
        max_support = Nmax / 10 - 1
        Ng = gen_cluster_sizes(G, max_support)
        data_pot = dgp_po_creg(Ng=Ng, G=G, tau_vec=tau_vec, gamma_vec=gamma_vec, n_treat=len(tau_vec))
        strata = form_strata_creg(data_pot, n_strata)
        strata_set = pd.DataFrame(strata)
        strata_set['S'] = strata_set.idxmax(axis=1) + 1
        pi_vec = [1 / (len(tau_vec) + 1)] * len(tau_vec)
        data_sim = dgp_obs_creg(data_pot, I_S=strata, pi_vec=pi_vec, n_treat=len(tau_vec))
        Y = data_sim['Y']
        D = data_sim['D']
        S = data_sim['S']
        if is_cov:
            X = data_sim['X']
        Ng = data_sim['Ng']
        G_id = data_sim['G_id']
        if is_cov:
            data_sim = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng, **{f'X{i+1}': X[:, i] for i in range(X.shape[1])}})
        else:
            data_sim = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng})
    else:
        data_pot = dgp_po_sreg(n=n, theta_vec=tau_vec, gamma_vec=gamma_vec, n_treat=len(tau_vec), is_cov=is_cov)
        strata = form_strata_sreg(data_pot, num_strata=n_strata)
        strata_set = pd.DataFrame(strata)
        strata_set['S'] = strata_set.idxmax(axis=1) + 1
        pi_vec = [1 / (len(tau_vec) + 1)] * len(tau_vec)
        data_sim = dgp_obs_sreg(data_pot, I_S=strata, pi_vec=pi_vec, n_treat=len(tau_vec), is_cov=is_cov)
        Y = data_sim['Y']
        D = data_sim['D']
        S = strata_set['S']
        if is_cov:
            X = np.array(data_sim['X'])
            data_sim = pd.DataFrame({'Y': Y, 'S': S, 'D': D, **{f'X{i+1}': X[:, i] for i in range(X.shape[1])}})
        else:
            data_sim = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
    
    return data_sim
