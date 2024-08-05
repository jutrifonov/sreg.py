import pandas as pd
import numpy as np
#-------------------------------------------------------------------
# %#     Auxiliary function providing the appropriate data.frame
# %#     for the subsequent iterative OLS estimation. Takes into
# %#     account the number of observations and creates indicators.
#-------------------------------------------------------------------
def subsample_ols_sreg(Y, S, D, X, s, d):
    # Ensure s and d are lists
    if not isinstance(s, (list, tuple, np.ndarray)):
        s = [s]
    if not isinstance(d, (list, tuple, np.ndarray)):
        d = [d]
    
    # Convert X to a numpy array
    X = np.array(X)
    
    # Create a DataFrame
    data = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, X.shape[1] + 1)])
    data = pd.concat([data, X_df], axis=1)
    
    # Filter data
    filtered_data = data[data['D'].isin(d) & data['S'].isin(s)]
    
    # Return the filtered data
    return filtered_data

def subsample_ols_creg(data, s, d):
    # Ensure s and d are scalar values (not lists)
    if isinstance(s, (list, tuple, np.ndarray)):
        s = s[0]
    if isinstance(d, (list, tuple, np.ndarray)):
        d = d[0]
    
    # Filter data
    filtered_data = data[(data['D'] == d) & (data['S'] == s)]
    
    # Return the filtered data
    return filtered_data

