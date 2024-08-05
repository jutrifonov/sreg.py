import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

from .subsample_filter import subsample_ols_sreg, subsample_ols_creg
#-------------------------------------------------------------------
# %#     Function that implements the OLS estimation of the
# %#     fully-saturated regression via lm() on the corresponding
# %#     subsamples generated via filter.ols.sreg()
#-------------------------------------------------------------------
def lm_iter_sreg(Y, S, D, X):
    # Initialize theta list
    theta_list = [np.full((max(S), X.shape[1]), np.nan) for _ in range(max(D) + 1)]

    # Iterate through d and s
    for d in range(max(D) + 1):
        for s in range(1, max(S) + 1):
            data_filtered = subsample_ols_sreg(Y, S, D, X, s, d)
            if data_filtered.empty:
                theta_list[d][s-1, :] = np.nan
                continue

            data_X = data_filtered.iloc[:, 3:3 + X.shape[1]]
            data_filtered_adj = pd.DataFrame({'Y': data_filtered['Y']}).join(data_X)

            try:
                X_const = add_constant(data_filtered_adj.drop(columns='Y'))
                model = OLS(data_filtered_adj['Y'], X_const).fit()
                theta_list[d][s-1, :] = model.params[1:].values
            except:
                theta_list[d][s-1, :] = np.nan

    return theta_list

def lm_iter_creg(Y, S, D, G_id, Ng, X):
    # Create working DataFrame
    if Ng is not None:
        working_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng})
        for i, col in enumerate(X.columns):
            working_df[col] = X.iloc[:, i]
    else:
        working_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id})
        #working_df = working_df.groupby('G_id').apply(lambda x: x.assign(Ng=len(x))).reset_index(drop=True)
        working_df['Ng'] = working_df.groupby('G_id')['G_id'].transform('count')
        for i, col in enumerate(X.columns):
            working_df[col] = X.iloc[:, i]
        
        
    
    # Aggregate Y by G_id
    Y_bar_g = working_df.groupby('G_id')['Y'].mean().reset_index()
    Y_bar_g.columns = ['G_id', 'Y_bar']
    
    # Create cl.lvl.data DataFrame
    cl_lvl_data = working_df.drop_duplicates(subset=['G_id', 'D', 'S', 'Ng'] + list(X.columns))
    cl_lvl_data_1 = pd.merge(Y_bar_g, cl_lvl_data, on='G_id')
    data = cl_lvl_data_1
    
    # Initialize theta list
    theta_list = [np.full((max(S), X.shape[1]), np.nan) for _ in range(max(D) + 1)]

    # Iterate through d and s
    for d in range(max(D) + 1):
        for s in range(1, max(S) + 1):
            data_filtered = subsample_ols_creg(data, s, d)
            if data_filtered.empty:
                theta_list[d][s-1, :] = np.nan
                continue

            data_X = data_filtered.iloc[:, 6:6 + X.shape[1]]
            data_filtered_adj = pd.DataFrame({'Y_bar_Ng': data_filtered['Y_bar'] * data_filtered['Ng']}).join(data_X)

            try:
                X_const = add_constant(data_filtered_adj.drop(columns='Y_bar_Ng'))
                model = OLS(data_filtered_adj['Y_bar_Ng'], X_const).fit()
                theta_list[d][s-1, :] = model.params[1:].values
            except:
                 [d][s-1, :] = np.nan

    return {
        "theta_list": theta_list,
        "cl_lvl_data": data
    }
