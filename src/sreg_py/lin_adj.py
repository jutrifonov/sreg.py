import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
#-------------------------------------------------------------------
# %#     Function that implements the calculation of \hat{\mu} --
# %#     i.e., calculates linear adjustments
def lin_adj_sreg(a, S, X, model):
    # Combine S and X into a DataFrame
    data = pd.DataFrame({'S': S})
    X_df = pd.DataFrame(X)
    data = pd.concat([data, X_df], axis=1)

    # Extract the theta matrix from the model
    theta_mtrx = model[a]

    # Match theta vectors to the S values
    theta_vec_mapped = theta_mtrx[S-1, :]

    # Calculate mu_hat
    mu_hat = np.einsum('ij,ij->i', X, theta_vec_mapped)
    
    return mu_hat

def lin_adj_creg(a, data, model):
    # Extract the X.data part of the data
    X_data = data.iloc[:, 6:]  # Select columns from the 6th to the last
    
    # Extract the theta matrix from the model
    theta_mtrx = model['theta_list'][a]
    
    # Match theta vectors to the S values
    theta_vec_mapped = theta_mtrx[data['S'].values - 1, :]
    
    # Calculate mu.hat
    mu_hat = np.einsum('ij,ij->i', X_data.values, theta_vec_mapped)
    
    return mu_hat
