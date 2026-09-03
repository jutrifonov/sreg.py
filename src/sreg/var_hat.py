import numpy as np
import pandas as pd
import warnings

from .pi_hat import pi_hat_sreg
from .lin_adj import lin_adj_sreg
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
            data = pd.DataFrame({'Y': np.asarray(Y), 'S': np.asarray(S), 'D': np.asarray(D)})
            X_df = pd.DataFrame(X).reset_index(drop=True)
            data = pd.concat([data, X_df], axis=1)
            data['pi'] = pi_hat_sreg(S, D)[:, d-1]
            data['pi_0'] = pi_hat_sreg(S, D, inverse=True)[:, 0]
            n = len(Y)
            data['A'] = np.where(D == d, 1, np.where(D == 0, 0, -999999))
            data['I'] = (data['A'] != -999999).astype(int)
            data['I_other'] = (data['A'] == -999999).astype(int)

            mu_hat_d = lin_adj_sreg(d, data['S'], data.iloc[:, 3:(3 + X.shape[1])], model)
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
            Xi_other = np.asarray(mu_hat_d-mu_hat_0,dtype=float)
            Xi_hat_other = Xi_other-pd.Series(Xi_other).groupby(data['S']).transform('mean').to_numpy()

            within=np.mean((data['A']==1)*Xi_hat_1**2+(data['A']==0)*Xi_hat_0**2+data['I_other']*Xi_hat_other**2)
            sigma_hat_sq = within + np.mean(Xi_hat_2 ** 2)
            if HC1:
                denom=n-(np.max(S)+np.max(D)*np.max(S))
                if denom <= 0:
                    warnings.warn("HC1 adjustment unstable or undefined due to degenerate strata-treatment structure; reverting to unadjusted estimator.",UserWarning,stacklevel=2)
                    var_vec[d-1]=sigma_hat_sq
                else:
                    var_vec[d-1]=within*(n/denom)+np.mean(Xi_hat_2**2)
            else:
                var_vec[d-1] = sigma_hat_sq
            n_vec[d-1] = n
        else:
            data = pd.DataFrame({'Y': np.asarray(Y), 'S': np.asarray(S), 'D': np.asarray(D)})
            data['pi'] = pi_hat_sreg(S, D)[:, d-1]
            data['pi_0'] = pi_hat_sreg(S, D, inverse=True)[:, 0]
            n = len(Y)
            data['A'] = np.where(D == d, 1, np.where(D == 0, 0, -999999))
            data['I'] = (data['A'] != -999999).astype(int)
            data['I_other'] = (data['A'] == -999999).astype(int)

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
                denom=n-(np.max(S)+np.max(D)*np.max(S))
                if denom <= 0:
                    warnings.warn("HC1 adjustment unstable or undefined due to degenerate strata-treatment structure; reverting to unadjusted estimator.",UserWarning,stacklevel=2)
                    var_vec[d-1]=sigma_hat_sq
                else:
                    within=np.mean(data['I']*(data['A']*Xi_hat_1**2+(1-data['A'])*Xi_hat_0**2))
                    var_vec[d-1]=within*(n/denom)+np.mean(Xi_hat_2**2)
            else:
                var_vec[d-1] = sigma_hat_sq
            n_vec[d-1] = n

    se_vec = np.sqrt(var_vec / n_vec)
    return se_vec

def as_var_creg(model=None, fit=None, HC1=False):
    result = np.zeros(len(fit['tau_hat']))
    for d in range(len(result)):
        ybar=np.asarray(fit['Y_bar_g']); ng=np.asarray(fit['Ng']); n=len(ybar)
        data=fit['data_list'][d].copy().reset_index(drop=True)
        if model is None: mu0=np.zeros(n); mud=np.zeros(n)
        else: mu0=np.asarray(fit['mu_hat'][d])[:,0]; mud=np.asarray(fit['mu_hat'][d])[:,1]
        pi=np.asarray(fit['pi_hat'][d]); pi0=np.asarray(fit['pi_hat_0'])
        total=ng*ybar; contrast=mud-mu0
        xt1=contrast+(total-mud)/pi; xt0=contrast-(total-mu0)/pi0
        strata=pd.Index(sorted(data.S.unique())); code=strata.get_indexer(data.S)
        active=data.A.ne(-999999)
        means1=pd.Series(xt1[active]).groupby([data.loc[active,'S'].to_numpy(),data.loc[active,'A'].to_numpy()]).mean()
        means0=pd.Series(xt0[active]).groupby([data.loc[active,'S'].to_numpy(),data.loc[active,'A'].to_numpy()]).mean()
        totals=pd.Series(total[active]).groupby([data.loc[active,'S'].to_numpy(),data.loc[active,'A'].to_numpy()]).mean()
        m1=np.array([means1.loc[(s,1)] for s in data.S]); m0=np.array([means0.loc[(s,0)] for s in data.S])
        t1=np.array([totals.loc[(s,1)] for s in data.S]); t0=np.array([totals.loc[(s,0)] for s in data.S])
        nbar=data.groupby('S').Ng.mean().reindex(data.S).to_numpy()
        correction=fit['tau_hat'][d]*(ng-nbar)
        x1=xt1-m1-correction; x0=xt0-m0-correction
        other=contrast-pd.Series(contrast).groupby(data.S).transform('mean').to_numpy()-correction
        x2=t1-t0-fit['tau_hat'][d]*nbar
        within=np.mean((data.A.eq(1))*x1**2+(data.A.eq(0))*x0**2+(data.A.eq(-999999))*other**2)
        denom=n-(data.S.max()+data.D.max()*data.S.max())
        if HC1 and denom <= 0:
            warnings.warn("HC1 adjustment unstable or undefined due to degenerate strata-treatment structure; reverting to unadjusted estimator.",UserWarning,stacklevel=2)
        factor=n/denom if HC1 and denom>0 else 1
        variance=(factor*within+np.mean(x2**2))/ng.mean()**2
        result[d]=np.sqrt(variance/n)
    return result
