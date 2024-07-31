import numpy as np
import pandas as pd
from scipy.stats import norm
#-------------------------------------------------------------------
# %#     The core function. It provides estimates of ATE, their s.e.,
# %#     calculates t-stats and corresponding p-values
#-------------------------------------------------------------------
def res_sreg(Y, S=None, D=None, X=None, HC1=True):
    n = len(Y)
    if S is None:
        S = np.ones(n)
    
    if X is not None:
        model = lm_iter_sreg(Y, S, D, X)
        tau_est = tau_hat_sreg(Y, S, D, X, model)
        se_rob = as_var_sreg(Y, S, D, X, model, tau_est, HC1)
        t_stat = tau_est / se_rob
        p_value = 2 * np.minimum(norm.cdf(t_stat), 1 - norm.cdf(t_stat))
        CI_left = tau_est - norm.ppf(0.975) * se_rob
        CI_right = tau_est + norm.ppf(0.975) * se_rob
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        data_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
        data_df = pd.concat([data_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        res_list = {
            "tau_hat": tau_est,
            "se_rob": se_rob,
            "t_stat": t_stat,
            "p_value": p_value,
            "as_CI": np.array([CI_left, CI_right]),
            "ols_iter": model,
            "CI_left": CI_left,
            "CI_right": CI_right,
            "data": data_df,
            #"data": pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'X': X}),
            "lin_adj": pd.DataFrame(X)
        }
    else:
        tau_est = tau_hat_sreg(Y, S, D, X=None, model=None)
        se_rob = as_var_sreg(Y, S, D, X=None, model=None, tau=tau_est, HC1=HC1)
        t_stat = tau_est / se_rob
        p_value = 2 * np.minimum(norm.cdf(t_stat), 1 - norm.cdf(t_stat))
        CI_left = tau_est - norm.ppf(0.975) * se_rob
        CI_right = tau_est + norm.ppf(0.975) * se_rob
        res_list = {
            "tau_hat": tau_est,
            "se_rob": se_rob,
            "t_stat": t_stat,
            "p_value": p_value,
            "as_CI": np.array([CI_left, CI_right]),
            "ols_iter": None,
            "CI_left": CI_left,
            "CI_right": CI_right,
            "data": pd.DataFrame({'Y': Y, 'S': S, 'D': D}),
            "lin_adj": None
        }
    
    class Sreg:
        def __init__(self, res_dict):
            self.__dict__.update(res_dict)
        def __repr__(self):
            return '\n'.join([f'{key}: {value}' for key, value in self.__dict__.items()])

    return Sreg(res_list)

# result=res_sreg(Y, S, D, X, HC1 = True)
# result=res_sreg(Y, S, D, X = None, HC1 = True)
# print(result)


def res_creg(Y, S, D, G_id, Ng, X, HC1=True):
    n = len(Y)

    if S is None:
        S = np.ones(n)
    
    if X is not None:
        df = pd.DataFrame({'G_id': G_id})
        X_df = pd.DataFrame(X)
        df = pd.concat([df, X_df], axis=1)
        
        if not check_cluster(df):
            X_0 = X
            df_mod = df.groupby('G_id').transform(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x)
            X = df_mod.iloc[:, 1:].values
        else:
            X_0 = X
        
        model = lm_iter_creg(Y, S, D, G_id, Ng, X)
        fit = tau_hat_creg(Y, S, D, G_id, Ng, X, model)
        tau_est = fit['tau_hat']
        se_rob = as_var_creg(model, fit, HC1)
        t_stat = tau_est / se_rob
        p_value = 2 * np.minimum(norm.cdf(t_stat), 1 - norm.cdf(t_stat))
        CI_left = tau_est - norm.ppf(0.975) * se_rob
        CI_right = tau_est + norm.ppf(0.975) * se_rob
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        data_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng})
        data_df = pd.concat([data_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

        if Ng is not None:
            res_list = {
                "tau_hat": tau_est,
                "se_rob": se_rob,
                "t_stat": t_stat,
                "p_value": p_value,
                "as_CI": np.array([CI_left, CI_right]),
                "ols_iter": model['theta_list'],
                "CI_left": CI_left,
                "CI_right": CI_right,
                "data": data_df,
                "lin_adj": pd.DataFrame(X_0)
            }
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            data_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id})
            data_df = pd.concat([data_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
            res_list = {
                "tau_hat": tau_est,
                "se_rob": se_rob,
                "t_stat": t_stat,
                "p_value": p_value,
                "as_CI": np.array([CI_left, CI_right]),
                "ols_iter": model['theta_list'],
                "CI_left": CI_left,
                "CI_right": CI_right,
                "data": data_df,
                "lin_adj": pd.DataFrame(X_0)
            }
    else:
        fit = tau_hat_creg(Y, S, D, G_id, Ng, X=None, model=None)
        tau_est = fit['tau_hat']
        se_rob = as_var_creg(model=None, fit=fit, HC1=HC1)
        t_stat = tau_est / se_rob
        p_value = 2 * np.minimum(norm.cdf(t_stat), 1 - norm.cdf(t_stat))
        CI_left = tau_est - norm.ppf(0.975) * se_rob
        CI_right = tau_est + norm.ppf(0.975) * se_rob
        
        if Ng is not None:
            res_list = {
                "tau_hat": tau_est,
                "se_rob": se_rob,
                "t_stat": t_stat,
                "p_value": p_value,
                "as_CI": np.array([CI_left, CI_right]),
                "ols_iter": None,
                "CI_left": CI_left,
                "CI_right": CI_right,
                "data": pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng}),
                "lin_adj": None
            }
        else:
            res_list = {
                "tau_hat": tau_est,
                "se_rob": se_rob,
                "t_stat": t_stat,
                "p_value": p_value,
                "as_CI": np.array([CI_left, CI_right]),
                "ols_iter": None,
                "CI_left": CI_left,
                "CI_right": CI_right,
                "data": pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id}),
                "lin_adj": None
            }
    class Creg:
        def __init__(self, res_dict):
            self.__dict__.update(res_dict)
        def __repr__(self):
            return '\n'.join([f'{key}: {value}' for key, value in self.__dict__.items()])
    
    return Creg(res_list)
    



#  result=res_creg(Y, S, D, G_id, Ng, X, HC1=False)
#  print(result)

#  result=res_creg(Y = Y, S = S, D = D, G_id = G_id, Ng = None, X = X, HC1=False)
#  print(result)

#  result=res_creg(Y = Y, S = S, D = D, G_id = G_id, Ng = None, X = None, HC1=False)
#  print(result)