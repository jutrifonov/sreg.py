import numpy as np
import pandas as pd
#-------------------------------------------------------------------
# %#     Function that implements \hat{\sigma^2} --
# %#     i.e. the variance estimator
#-------------------------------------------------------------------
def as_var_sreg(Y, S, D, X=None, model=None, tau=None, HC1=True):
    max_D = np.max(D)
    var_vec = np.zeros(max_D)
    n_vec = np.zeros(max_D)

    for d in range(1, max_D + 1):
        if X is not None:
            data = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
            X_df = pd.DataFrame(X)
            data = pd.concat([data, X_df], axis=1)
            data['pi'] = pi_hat_sreg(S, D)[:, d-1]
            data['pi_0'] = pi_hat_sreg(S, D, inverse=True)[:, 0]
            n = len(Y)
            data['A'] = np.where(D == d, 1, np.where(D == 0, 0, -999999))
            data['I'] = (data['A'] != -999999).astype(int)

            mu_hat_d = lin_adj_sreg(d, data['S'], data.iloc[:, 3:(3 + X.shape[1])], model)
             #lin_adj_sreg(d, data['S'], data.iloc[:, 3:(3+X.shape[1])], model)
            mu_hat_0 = lin_adj_sreg(0, data['S'], data.iloc[:, 3:(3 + X.shape[1])], model)

            Xi_tilde_1 = (mu_hat_d - mu_hat_0) + (data['Y'] - mu_hat_d) / data['pi']
            Xi_tilde_0 = (mu_hat_d - mu_hat_0) - (data['Y'] - mu_hat_0) / data['pi_0']

            data = pd.concat([data, pd.DataFrame({'Xi_tilde_1': Xi_tilde_1, 'Xi_tilde_0': Xi_tilde_0, 
                                                 'Y_tau_D': data['Y'] - tau[d-1] * data['A'] * data['I']})], axis=1)

            count_Xi_1 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_1=('Xi_tilde_1', 'mean')).reset_index()
            count_Xi_0 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_0=('Xi_tilde_0', 'mean')).reset_index()
            count_Y = data[data['A'] != -999999].groupby(['S', 'A']).agg(Y_tau=('Y_tau_D', 'mean')).reset_index()

            j = count_Xi_1.merge(count_Xi_0, on=['S', 'A']).merge(count_Y, on=['S', 'A'])

            Xi_tilde_1_all = j.pivot(index='S', columns='A', values='Xi_mean_1').fillna(0)
            Xi_tilde_0_all = j.pivot(index='S', columns='A', values='Xi_mean_0').fillna(0)
            Y_tau_D_all = j.pivot(index='S', columns='A', values='Y_tau').fillna(0)

            Xi_tilde_1_mean = Xi_tilde_1_all.values
            Xi_tilde_0_mean = Xi_tilde_0_all.values
            Y_tau_D_mean = Y_tau_D_all.values

            Xi_1_mean = Xi_tilde_1_mean[S - 1, 1]
            Xi_0_mean = Xi_tilde_0_mean[S - 1, 0]
            Y_tau_D_1_mean = Y_tau_D_mean[S - 1, 1]
            Y_tau_D_0_mean = Y_tau_D_mean[S - 1, 0]

            Xi_hat_1 = Xi_tilde_1 - Xi_1_mean
            Xi_hat_0 = Xi_tilde_0 - Xi_0_mean
            Xi_hat_2 = Y_tau_D_1_mean - Y_tau_D_0_mean

            sigma_hat_sq = np.mean(data['I'] * (data['A'] * (Xi_hat_1 ** 2) + (1 - data['A']) * (Xi_hat_0 ** 2)) + Xi_hat_2 ** 2)
            if HC1:
                var_vec[d-1] = (np.mean(data['I'] * (data['A'] * (Xi_hat_1 ** 2) + (1 - data['A']) * (Xi_hat_0 ** 2))) * 
                                (n / (n - (np.max(S) + np.max(D) * np.max(S)))) + np.mean(Xi_hat_2 ** 2))
            else:
                var_vec[d-1] = sigma_hat_sq
            n_vec[d-1] = n
        else:
            data = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
            data['pi'] = pi_hat_sreg(S, D)[:, d-1]
            data['pi_0'] = pi_hat_sreg(S, D, inverse=True)[:, 0]
            n = len(Y)
            data['A'] = np.where(D == d, 1, np.where(D == 0, 0, -999999))
            data['I'] = (data['A'] != -999999).astype(int)

            mu_hat_d = 0
            mu_hat_0 = 0

            Xi_tilde_1 = (mu_hat_d - mu_hat_0) + (data['Y'] - mu_hat_d) / data['pi']
            Xi_tilde_0 = (mu_hat_d - mu_hat_0) - (data['Y'] - mu_hat_0) / data['pi_0']

            data = pd.concat([data, pd.DataFrame({'Xi_tilde_1': Xi_tilde_1, 'Xi_tilde_0': Xi_tilde_0, 
                                                 'Y_tau_D': data['Y'] - tau[d-1] * data['A'] * data['I']})], axis=1)

            count_Xi_1 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_1=('Xi_tilde_1', 'mean')).reset_index()
            count_Xi_0 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_0=('Xi_tilde_0', 'mean')).reset_index()
            count_Y = data[data['A'] != -999999].groupby(['S', 'A']).agg(Y_tau=('Y_tau_D', 'mean')).reset_index()

            j = count_Xi_1.merge(count_Xi_0, on=['S', 'A']).merge(count_Y, on=['S', 'A'])

            Xi_tilde_1_all = j.pivot(index='S', columns='A', values='Xi_mean_1').fillna(0)
            Xi_tilde_0_all = j.pivot(index='S', columns='A', values='Xi_mean_0').fillna(0)
            Y_tau_D_all = j.pivot(index='S', columns='A', values='Y_tau').fillna(0)

            Xi_tilde_1_mean = Xi_tilde_1_all.values
            Xi_tilde_0_mean = Xi_tilde_0_all.values
            Y_tau_D_mean = Y_tau_D_all.values

            Xi_1_mean = Xi_tilde_1_mean[S - 1, 1]
            Xi_0_mean = Xi_tilde_0_mean[S - 1, 0]
            Y_tau_D_1_mean = Y_tau_D_mean[S - 1, 1]
            Y_tau_D_0_mean = Y_tau_D_mean[S - 1, 0]

            Xi_hat_1 = Xi_tilde_1 - Xi_1_mean
            Xi_hat_0 = Xi_tilde_0 - Xi_0_mean
            Xi_hat_2 = Y_tau_D_1_mean - Y_tau_D_0_mean

            sigma_hat_sq = np.mean(data['I'] * (data['A'] * (Xi_hat_1 ** 2) + (1 - data['A']) * (Xi_hat_0 ** 2)) + Xi_hat_2 ** 2)
            if HC1:
                var_vec[d-1] = (np.mean(data['I'] * (data['A'] * (Xi_hat_1 ** 2) + (1 - data['A']) * (Xi_hat_0 ** 2))) * 
                                (n / (n - (np.max(S) + np.max(D) * np.max(S)))) + np.mean(Xi_hat_2 ** 2))
            else:
                var_vec[d-1] = sigma_hat_sq
            n_vec[d-1] = n

    se_vec = np.sqrt(var_vec / n_vec)
    return se_vec



# model = lm_iter_sreg(Y, S, D, X)
# tau = tau_hat_sreg(Y, S, D, X, model)

# as_var_sreg(Y, S, D, X, model=model, tau=tau, HC1=True)


# tau = tau_hat_sreg(Y, S, D)

# as_var_sreg(Y, S, D, X = None, model=None, tau=tau, HC1=True)




# data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg.py/src/sreg.py/data_cl.csv")
# print(data.head())
# Y = data['Y']
# D = data['D']
# S = data['S']
# G_id = data['G.id']
# Ng = data['Ng']
# X = data[['x_1', 'x_2']]




def as_var_creg(model=None, fit=None, HC1=False):
    var_vec = np.zeros(len(fit['tau_hat']))
    n_vec = np.zeros(len(fit['tau_hat']))

    if model is not None:
        for d in range(len(fit['tau_hat'])):
            Y_bar_g = fit['Y_bar_g']
            Ng = fit['Ng']
            mu_hat_0 = fit['mu_hat'][d][:, 0]
            mu_hat_d = fit['mu_hat'][d][:, 1]
            tau_est = fit['tau_hat']
            pi_hat = fit['pi_hat'][d]
            pi_hat_0 = fit['pi_hat_0']
            data = fit['data_list'][d]
            n = len(Y_bar_g)

            Xi_tilde_1 = (mu_hat_d - mu_hat_0) + (Ng * Y_bar_g - mu_hat_d) / pi_hat
            Xi_tilde_0 = (mu_hat_d - mu_hat_0) - (Ng * Y_bar_g - mu_hat_0) / pi_hat_0

            data = pd.concat([data, pd.DataFrame({'Xi_tilde_1': Xi_tilde_1, 'Xi_tilde_0': Xi_tilde_0, 'Y_Ng': Y_bar_g * Ng})], axis=1)

            count_Xi_1 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_1=('Xi_tilde_1', 'mean')).reset_index()
            count_Xi_0 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_0=('Xi_tilde_0', 'mean')).reset_index()
            count_Y = data[data['A'] != -999999].groupby(['S', 'A']).agg(Y_bar=('Y_Ng', 'mean')).reset_index()
            count_Ng = data.groupby('S').agg(Ng_bar=('Ng', 'mean')).reset_index()

            j = count_Xi_1.merge(count_Xi_0, on=['S', 'A']).merge(count_Y, on=['S', 'A']).merge(count_Ng, on='S')

            Xi_tilde_1_all = j.pivot(index='S', columns='A', values='Xi_mean_1').fillna(0)
            Xi_tilde_0_all = j.pivot(index='S', columns='A', values='Xi_mean_0').fillna(0)
            Y_Ng_all = j.pivot(index='S', columns='A', values='Y_bar').fillna(0)
            Ng_bar_all = j.pivot(index='S', columns='A', values='Ng_bar').fillna(0)

            Xi_tilde_1_mean = Xi_tilde_1_all.values
            Xi_tilde_0_mean = Xi_tilde_0_all.values
            Y_Ng_mean = Y_Ng_all.values
            Ng_bar_mean = Ng_bar_all.values

            Xi_1_mean = Xi_tilde_1_mean[data['S'] - 1, 1]
            Xi_0_mean = Xi_tilde_0_mean[data['S'] - 1, 0]
            Y_g_bar_cl_1 = Y_Ng_mean[data['S'] - 1, 1]
            Y_g_bar_cl_0 = Y_Ng_mean[data['S'] - 1, 0]
            N_g_bar_cl = Ng_bar_mean[data['S'] - 1, 0]

            Xi_hat_1 = Xi_tilde_1 - Xi_1_mean - tau_est[d] * (Ng - N_g_bar_cl)
            Xi_hat_0 = Xi_tilde_0 - Xi_0_mean - tau_est[d] * (Ng - N_g_bar_cl)
            Xi_hat_2 = Y_g_bar_cl_1 - Y_g_bar_cl_0 - tau_est[d] * N_g_bar_cl

            sigma_hat_sq = np.mean(data['I'] * (data['A'] * Xi_hat_1 ** 2 + (1 - data['A']) * Xi_hat_0 ** 2) + Xi_hat_2 ** 2) / (np.mean(Ng)) ** 2

            if HC1:
                var_vec[d] = ((np.mean(data['I'] * (data['A'] * Xi_hat_1 ** 2 + (1 - data['A']) * Xi_hat_0 ** 2))) * 
                              (n / (n - (np.max(data['S']) + np.max(data['D']) * np.max(data['S'])))) + np.mean(Xi_hat_2 ** 2)) / (np.mean(Ng)) ** 2
            else:
                var_vec[d] = sigma_hat_sq
            n_vec[d] = n
    else:
        for d in range(len(fit['tau_hat'])):
            Y_bar_g = fit['Y_bar_g']
            Ng = fit['Ng']
            tau_est = fit['tau_hat']
            pi_hat = fit['pi_hat'][d]
            pi_hat_0 = fit['pi_hat_0']
            data = fit['data_list'][d]
            n = len(Y_bar_g)

            mu_hat_0 = 0
            mu_hat_d = 0

            Xi_tilde_1 = (mu_hat_d - mu_hat_0) + (Ng * Y_bar_g - mu_hat_d) / pi_hat
            Xi_tilde_0 = (mu_hat_d - mu_hat_0) - (Ng * Y_bar_g - mu_hat_0) / pi_hat_0

            data = pd.concat([data, pd.DataFrame({'Xi_tilde_1': Xi_tilde_1, 'Xi_tilde_0': Xi_tilde_0, 'Y_Ng': Y_bar_g * Ng})], axis=1)

            count_Xi_1 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_1=('Xi_tilde_1', 'mean')).reset_index()
            count_Xi_0 = data[data['A'] != -999999].groupby(['S', 'A']).agg(Xi_mean_0=('Xi_tilde_0', 'mean')).reset_index()
            count_Y = data[data['A'] != -999999].groupby(['S', 'A']).agg(Y_bar=('Y_Ng', 'mean')).reset_index()
            count_Ng = data.groupby('S').agg(Ng_bar=('Ng', 'mean')).reset_index()

            j = count_Xi_1.merge(count_Xi_0, on=['S', 'A']).merge(count_Y, on=['S', 'A']).merge(count_Ng, on='S')

            Xi_tilde_1_all = j.pivot(index='S', columns='A', values='Xi_mean_1').fillna(0)
            Xi_tilde_0_all = j.pivot(index='S', columns='A', values='Xi_mean_0').fillna(0)
            Y_Ng_all = j.pivot(index='S', columns='A', values='Y_bar').fillna(0)
            Ng_bar_all = j.pivot(index='S', columns='A', values='Ng_bar').fillna(0)

            Xi_tilde_1_mean = Xi_tilde_1_all.values
            Xi_tilde_0_mean = Xi_tilde_0_all.values
            Y_Ng_mean = Y_Ng_all.values
            Ng_bar_mean = Ng_bar_all.values

            Xi_1_mean = Xi_tilde_1_mean[data['S'] - 1, 1]
            Xi_0_mean = Xi_tilde_0_mean[data['S'] - 1, 0]
            Y_g_bar_cl_1 = Y_Ng_mean[data['S'] - 1, 1]
            Y_g_bar_cl_0 = Y_Ng_mean[data['S'] - 1, 0]
            N_g_bar_cl = Ng_bar_mean[data['S'] - 1, 0]

            Xi_hat_1 = Xi_tilde_1 - Xi_1_mean - tau_est[d] * (Ng - N_g_bar_cl)
            Xi_hat_0 = Xi_tilde_0 - Xi_0_mean - tau_est[d] * (Ng - N_g_bar_cl)
            Xi_hat_2 = Y_g_bar_cl_1 - Y_g_bar_cl_0 - tau_est[d] * N_g_bar_cl

            sigma_hat_sq = np.mean(data['I'] * (data['A'] * Xi_hat_1 ** 2 + (1 - data['A']) * Xi_hat_0 ** 2) + Xi_hat_2 ** 2) / (np.mean(Ng)) ** 2

            if HC1:
                var_vec[d] = ((np.mean(data['I'] * (data['A'] * Xi_hat_1 ** 2 + (1 - data['A']) * Xi_hat_0 ** 2))) * 
                              (n / (n - (np.max(data['S']) + np.max(data['D']) * np.max(data['S'])))) + np.mean(Xi_hat_2 ** 2)) / (np.mean(Ng)) ** 2
            else:
                var_vec[d] = sigma_hat_sq
            n_vec[d] = n
    se_vec = np.sqrt(var_vec / n_vec)
    return se_vec


# model = lm_iter_creg(Y, S, D, G_id, Ng, X)
# fit = tau_hat_creg(Y, S, D, G_id, Ng, X, model)
# fit['tau_hat']
# as_var_creg(model, fit, HC1 = False)


# fit = tau_hat_creg(Y, S, D, G_id, Ng)
# fit['tau_hat']
# as_var_creg(model = None, fit = fit, HC1 = False)

# fit = tau_hat_creg(Y, S, D, G_id, Ng = None)
# fit['tau_hat']
# as_var_creg(model = None, fit = fit, HC1 = False)



