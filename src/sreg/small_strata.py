"""Small-strata and mixed-design estimators.

This module is a direct numerical port of the estimators introduced in sreg R
2.1.0.  It deliberately uses NumPy/Pandas only; R is used by the test suite as
an oracle, never at runtime.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm


def classify_strata(data, *, cluster=False, small_strata=True, k=None,
                    warn=True, keep_size=False):
    counts = (data[["S", "G_id"]].drop_duplicates().groupby("S").size()
              if cluster else data.groupby("S").size())
    if not small_strata:
        frequencies = counts.value_counts(normalize=True)
        eligible = frequencies[(frequencies.index == k) if k is not None
                               else (frequencies.index <= 3)]
        if warn and (eligible >= .25).any():
            warnings.warn("At least 25% of strata are small, but small_strata=False. "
                          "Setting small_strata=True is required for valid mixed-design standard errors.",
                          UserWarning, stacklevel=2)
        return data.copy()
    sizes = counts.unique()
    if len(sizes) == 1:
        if k is not None and sizes[0] != k:
            raise ValueError(f"The supplied small-stratum size k = {k} does not match "
                             f"the observed stratum size of {sizes[0]}.")
        mapping = pd.DataFrame({"S": counts.index, "size": counts.values,
                                "stratum_type": "small"})
    else:
        frequency = counts.value_counts(normalize=True)
        eligible = frequency[(frequency.index == k) if k is not None
                             else (frequency.index <= 3)]
        eligible = eligible[eligible >= .25]
        if eligible.empty:
            raise ValueError("Invalid input: either all strata are large or too few strata "
                             "qualify as small; set small_strata=False.")
        modal = eligible.index[0]
        mapping = pd.DataFrame({"S": counts.index, "size": counts.values})
        mapping["stratum_type"] = np.where(mapping["size"] == modal, "small", "big")
        if warn and (mapping.stratum_type == "big").any():
            warnings.warn(f"Mixed design detected: at least 25% of strata have size k = {modal}; "
                          "weighted estimators will be used.", UserWarning, stacklevel=2)
    if not keep_size:
        mapping = mapping.drop(columns="size")
    return data.merge(mapping, on="S", how="left")


def _result(tau, se, data, *, beta=None, lin_adj=None, **extra):
    tau, se = np.asarray(tau, float), np.asarray(se, float)
    ci_left, ci_right = tau - norm.ppf(.975) * se, tau + norm.ppf(.975) * se
    out = {"tau_hat": tau, "se_rob": se, "t_stat": tau / se,
           "p_value": 2 * norm.sf(np.abs(tau / se)),
           "as_CI": np.r_[ci_left, ci_right], "CI_left": ci_left,
           "CI_right": ci_right, "beta_hat": beta, "data": data,
           "lin_adj": lin_adj}
    out.update(extra)
    return out


def _individual_fit(data, xcols):
    treatments = sorted(set(data.D) - {0})
    tau, betas = [], []
    x = data[xcols].to_numpy(float) if xcols else None
    for d in treatments:
        beta = None
        if xcols:
            yd, xd = [], []
            for _, group in data.groupby("S", sort=False):
                yd.append(group.loc[group.D == d, "Y"].mean() - group.loc[group.D == 0, "Y"].mean())
                xd.append(group.loc[group.D == d, xcols].mean().to_numpy() -
                          group.loc[group.D == 0, xcols].mean().to_numpy())
            design = np.column_stack([np.ones(len(yd)), np.asarray(xd)])
            beta = np.linalg.lstsq(design, yd, rcond=None)[0][1:]
            centered = x - x.mean(axis=0)
            adjustment = (centered[data.D.to_numpy() == d].mean(axis=0) -
                          centered[data.D.to_numpy() == 0].mean(axis=0)) @ beta
        else:
            adjustment = 0
        tau.append(data.loc[data.D == d, "Y"].mean() -
                   data.loc[data.D == 0, "Y"].mean() - adjustment)
        betas.append(beta)
    return np.asarray(tau), None if not xcols else np.vstack(betas)


def _individual_variance(data, xcols, tau, beta, hc1):
    n_blocks = data.S.nunique()
    if n_blocks % 2:
        raise ValueError("The paired-strata variance estimator requires an even number of strata.")
    arms, values = sorted(data.D.unique()), []
    for pos, d in enumerate(arms[1:]):
        if xcols:
            centered = data[xcols].to_numpy(float) - data[xcols].mean().to_numpy()
            ya = data.Y.to_numpy() - centered @ beta[pos]
        else:
            ya = data.Y.to_numpy()
        frame = pd.DataFrame({"Y": ya, "D": data.D.to_numpy(), "S": data.S.to_numpy()})
        l, q = (frame.D == d).sum() / n_blocks, (frame.D == 0).sum() / n_blocks
        gamma1, gamma0 = frame.loc[frame.D == d, "Y"].mean(), frame.loc[frame.D == 0, "Y"].mean()
        st = frame.Y.where(frame.D == d, 0).groupby(frame.S).sum().to_numpy()
        sc = frame.Y.where(frame.D == 0, 0).groupby(frame.S).sum().to_numpy()
        rho11 = (2 / n_blocks) * np.sum(st[::2] * st[1::2]) / l**2
        rho00 = (2 / n_blocks) * np.sum(sc[::2] * sc[1::2]) / q**2
        rho10 = np.mean(st * sc) / (l * q)
        sigma1 = np.sum((frame.Y - gamma1)**2 * (frame.D == d)) / (n_blocks * l)
        sigma0 = np.sum((frame.Y - gamma0)**2 * (frame.D == 0)) / (n_blocks * q)
        v11, v10 = sigma1 - (rho11 - gamma1**2), sigma0 - (rho00 - gamma0**2)
        factor = n_blocks / (n_blocks - (len(xcols) + 1)) if hc1 and xcols else 1
        variance = factor * v11 / (l / data.groupby("S").size().iloc[0])
        variance += factor * v10 / (q / data.groupby("S").size().iloc[0])
        variance += rho11 - gamma1**2 + rho00 - gamma0**2 - 2*(rho10-gamma1*gamma0)
        values.append(variance / len(data))
    return np.sqrt(np.maximum(values, 0))


def res_sreg_small(Y, S, D, X=None, HC1=True):
    frame = pd.DataFrame({"Y": Y, "S": S, "D": D}).reset_index(drop=True)
    xcols = []
    if X is not None:
        xdf = pd.DataFrame(X).reset_index(drop=True)
        xdf.columns = [str(c) for c in xdf.columns]
        frame = pd.concat([frame, xdf], axis=1); xcols = list(xdf.columns)
    tau, beta = _individual_fit(frame, xcols)
    se = _individual_variance(frame, xcols, tau, beta, HC1)
    return _result(tau, se, frame, beta=beta, lin_adj=frame[xcols] if xcols else None)


def _cluster_frame(Y, S, D, G_id, Ng, X):
    raw = pd.DataFrame({"Y": Y, "S": S, "D": D, "G_id": G_id})
    raw["Ng"] = raw.groupby("G_id").Y.transform("size") if Ng is None else np.asarray(Ng)
    base = raw.groupby("G_id", sort=False).agg(Y_bar=("Y", "mean"), S=("S", "first"),
                                                D=("D", "first"), Ng=("Ng", "first")).reset_index()
    xcols = []
    if X is not None:
        xdf = pd.DataFrame(X); xcols = [f"x{i}" for i in range(xdf.shape[1])]; xdf.columns=xcols
        xdf["G_id"] = np.asarray(G_id)
        base = base.merge(xdf.groupby("G_id", sort=False)[xcols].mean().reset_index(), on="G_id")
    base["T"] = base.Ng * base.Y_bar
    return base, xcols


def res_creg_small(Y, S, D, G_id, Ng=None, X=None, HC1=True):
    data, xcols = _cluster_frame(Y, S, D, G_id, Ng, X)
    raw=pd.DataFrame({'Y':np.asarray(Y),'S':np.asarray(S),'D':np.asarray(D),'G_id':np.asarray(G_id)})
    raw['Ng']=raw.groupby('G_id').Y.transform('size') if Ng is None else np.asarray(Ng)
    if X is not None:
        original_x=pd.DataFrame(X).reset_index(drop=True); raw=pd.concat([raw,original_x],axis=1)
    proxy = data.rename(columns={"T": "Y"})
    tau_q, beta = _individual_fit(proxy, xcols)
    nbar = data.Ng.mean(); tau = tau_q / nbar
    # Port the R multi-arm linearization using the expanded cluster outcome.
    arms, n_blocks, variances = sorted(data.D.unique()), data.S.nunique(), []
    if n_blocks % 2: raise ValueError("The paired-strata variance estimator requires an even number of strata.")
    pi = data.D.value_counts(normalize=True)
    for pos, d in enumerate(arms[1:]):
        residual = data["T"].to_numpy(float).copy()
        K = 1
        if xcols:
            xm = data[xcols].to_numpy(float); residual -= (xm-xm.mean(0)) @ beta[pos]; K=len(xcols)+1
        W = -tau[pos] * data.D.map(pi).to_numpy() * data.Ng.to_numpy() / nbar
        W[data.D.to_numpy()==d] += residual[data.D.to_numpy()==d]/nbar
        W[data.D.to_numpy()==0] -= residual[data.D.to_numpy()==0]/nbar
        gamma={}; sigma={}; sums={}; karm={}
        for arm in arms:
            mask=data.D.to_numpy()==arm; gamma[arm]=W[mask].mean(); sigma[arm]=np.mean((W[mask]-gamma[arm])**2)
            sums[arm]=pd.Series(np.where(mask,W,0)).groupby(data.S.reset_index(drop=True)).sum().to_numpy()
            karm[arm]=mask.sum()/n_blocks
        rho=np.empty((len(arms),len(arms)))
        for i,a in enumerate(arms): rho[i,i]=(2/n_blocks)*np.sum(sums[a][::2]*sums[a][1::2])/karm[a]**2
        for i,a in enumerate(arms):
            for j,b in enumerate(arms[i+1:],i+1): rho[i,j]=rho[j,i]=np.mean(sums[a]*sums[b])/(karm[a]*karm[b])
        g=np.array([gamma[a] for a in arms]); v2=rho-np.outer(g,g)
        v1=np.array([sigma[a] for a in arms])-np.diag(v2); hc=n_blocks/(n_blocks-K) if HC1 else 1
        variances.append((np.sum(hc*v1/np.array([pi[a] for a in arms]))+v2.sum())/data.shape[0])
    return _result(tau, np.sqrt(np.maximum(variances,0)), raw, beta=beta,
                   lin_adj=original_x if X is not None else None)


def combine_mixed(small, big, p_small, share_variance=0.0, total_units=None):
    p_big = 1-p_small; tau=p_small*small["tau_hat"]+p_big*big["tau_hat"]
    cross = share_variance*(small["tau_hat"]-big["tau_hat"])**2
    if total_units is not None: cross /= total_units
    se=np.sqrt(p_small**2*small["se_rob"]**2+p_big**2*big["se_rob"]**2+cross)
    return tau,se
