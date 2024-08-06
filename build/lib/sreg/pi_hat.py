import pandas as pd
import numpy as np

#-------------------------------------------------------------------
# %#     Function that implements the calculation of \hat{\pi} --
# %#     i.e., calculates the proportions assigned to treatments
#-------------------------------------------------------------------
def pi_hat_sreg(S, D, inverse=False):
    # Combine S and D into a DataFrame
    data = pd.DataFrame({'S': S, 'D': D})

    # Count occurrences of S, D pairs and S values
    counts = data.groupby(['S', 'D']).size().reset_index(name='n').sort_values(by=['S', 'D'])
    scount = data.groupby('S').size().reset_index(name='ns')

    # Merge counts with scount
    j = counts.merge(scount, on='S')
    j['pi_hat'] = j['n'] / j['ns']

    # Pivot the DataFrame to get pi_hat values
    pi_hat_all = j.pivot(index='S', columns='D', values='pi_hat').fillna(0).reset_index()
    
    # Ensure columns are ordered correctly (S, 0, 1, 2, ...)
    columns_order = ['S'] + sorted(pi_hat_all.columns[1:], key=int)
    pi_hat_all = pi_hat_all[columns_order]
 
    if inverse:
        n_repeat = data['D'].max()
        ret_df = np.tile(pi_hat_all[columns_order[1]].values.reshape(-1, 1), n_repeat)

    else:
        ret_df = pi_hat_all.iloc[:, 2:].values  # Select columns starting from the second one

    # Create a mapping from S values to their corresponding indices in the DataFrame
    s_to_index = {s: i for i, s in enumerate(pi_hat_all['S'])}

    # Convert to matrix and return the specific rows for S
    s_indices = [s_to_index[s] for s in S]
    return ret_df[s_indices, :]

#-------------------------------------------------------------------
def pi_hat_creg(S, D, inverse=False):
#-------------------------------------------------------------------
    # Combine S and D into a DataFrame
    data = pd.DataFrame({'S': S, 'D': D})

    # Count occurrences of S, D pairs and S values
    counts = data.groupby(['S', 'D']).size().reset_index(name='n').sort_values(by=['S', 'D'])
    scount = data.groupby('S').size().reset_index(name='ns')

    # Merge counts with scount
    j = counts.merge(scount, on='S')
    j['pi_hat'] = j['n'] / j['ns']

    # Pivot the DataFrame to get pi_hat values
    pi_hat_all = j.pivot(index='S', columns='D', values='pi_hat').fillna(0).reset_index()
    
    # Ensure columns are ordered correctly (S, 0, 1, 2, ...)
    columns_order = ['S'] + sorted(pi_hat_all.columns[1:], key=int)
    pi_hat_all = pi_hat_all[columns_order]
 
    if inverse:
        n_repeat = data['D'].max()
        ret_df = np.tile(pi_hat_all[columns_order[1]].values.reshape(-1, 1), n_repeat)
    else:
        ret_df = pi_hat_all.iloc[:, 2:].values  # Select columns starting from the third one

    # Create a mapping from S values to their corresponding indices in the DataFrame
    s_to_index = {s: i for i, s in enumerate(pi_hat_all['S'])}

    # Convert to matrix and return the specific rows for S
    s_indices = [s_to_index[s] for s in S]
    return ret_df[s_indices, :]
