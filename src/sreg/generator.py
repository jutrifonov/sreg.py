"""R-2.1-compatible design generation helpers."""
import numpy as np
import pandas as pd


def _validate(n, tau, k, treat_sizes):
    if isinstance(n, bool) or int(n) != n or n <= 0: raise ValueError("n must be a positive integer.")
    if isinstance(k, bool) or int(k) != k or k <= 0: raise ValueError("k must be a positive integer.")
    ts=np.asarray(treat_sizes)
    if len(ts)!=len(tau)+1 or np.any(ts<0) or np.any(ts!=ts.astype(int)) or ts.sum()!=k:
        raise ValueError("treat_sizes must be a nonnegative integer vector of length len(tau_vec) + 1 that sums to k.")


def generate(n, Nmax=50, n_strata=10, tau_vec=(0,), gamma_vec=(.4,.2,1),
             cluster=True, is_cov=True, small_strata=False, k=3,
             treat_sizes=None, mixed_strata=False, n_small=None,
             allocation_probs=None, stratum_effects=None,
             treatment_effects_by_stratum=None, rng=None):
    rng=np.random.default_rng(rng); tau=np.atleast_1d(tau_vec).astype(float)
    gamma=np.atleast_1d(gamma_vec).astype(float)
    if gamma.shape != (3,) or not np.isfinite(gamma).all():
        raise ValueError("gamma_vec must be a finite numeric vector of length 3.")
    if mixed_strata:
        if treat_sizes is None:
            base=k//(len(tau)+1); remainder=k%(len(tau)+1)
            treat_sizes=tuple(base+(i<remainder) for i in range(len(tau)+1))
        if n_small is None: n_small=int((n/2)//k*k)
        _validate(n,tau,k,treat_sizes)
        if not 0<n_small<n or int(n_small)!=n_small: raise ValueError("n_small must be a positive integer smaller than n.")
        if n_small%k: raise ValueError("n_small must be divisible by k.")
        if n-n_small<=n_strata*k: raise ValueError("The large-strata component must contain more than k units per stratum on average.")
        if n_small/k<=n_strata: raise ValueError("The mixed design must contain more small strata than large strata.")
        a=generate(n_small,Nmax,n_strata,tau,gamma_vec,cluster,is_cov,True,k,treat_sizes,rng=rng)
        b=generate(n-n_small,Nmax,n_strata,tau,gamma_vec,cluster,is_cov,False,k,treat_sizes,rng=rng)
        b.S += a.S.max()
        if cluster: b.G_id += a.G_id.max()
        return pd.concat([a,b],ignore_index=True,sort=False)
    custom=any(x is not None for x in (allocation_probs,stratum_effects,treatment_effects_by_stratum))
    if custom and (cluster or small_strata or mixed_strata):
        raise ValueError("Custom large-strata arguments are supported only for individual-level large-strata designs.")
    if small_strata:
        # R's non-mixed formal default is c(1, 1, 1). Unlike mixed designs,
        # it is not derived from the number of arms when omitted.
        if treat_sizes is None: treat_sizes=(1,1,1)
        _validate(n,tau,k,treat_sizes)
    arms=np.arange(len(tau)+1); effects=np.r_[0,tau]
    if cluster:
        G=int(n)
        max_support=Nmax/10-1
        if max_support < 0 or int(max_support) != max_support:
            raise ValueError("Nmax must be a positive multiple of 10.")
        beta_p=rng.beta(1,1,G)
        sizes=10*(rng.binomial(int(max_support),beta_p)+1)
        z=(rng.beta(2,2,G)-.5)*np.sqrt(20)
        x1=rng.normal(5,2,G); x2=rng.normal(2,1,G)
        x1_std=(x1-5)/2; x2_std=x2-2
        baseline=gamma[0]*z+gamma[1]*x1_std+gamma[2]*x2_std
        order=np.argsort(z)
        if small_strata:
            if G%k: raise ValueError("n must be divisible by k for small-strata designs.")
            S=np.empty(G,int); D=np.empty(G,int)
            assignment=np.repeat(arms,np.asarray(treat_sizes,int))
            for j,idx in enumerate(order.reshape(-1,k),1): S[idx]=j; D[idx]=rng.permutation(assignment)
        else:
            bounds=np.linspace(z.min(),z.max(),int(n_strata)+1)
            S=np.clip(np.digitize(z,bounds[1:-1],right=True)+1,1,int(n_strata))
            D=np.empty(G,int)
            for s in range(1,int(n_strata)+1):
                idx=np.flatnonzero(S==s)
                active=np.full(len(tau),int(np.floor(len(idx)/(len(tau)+1))))
                control=len(idx)-active.sum()
                D[idx]=rng.permutation(np.repeat(arms,np.r_[control,active]))
        rows=[]
        for g in range(G):
            error_sd=1 if D[g]==0 else np.sqrt(2)
            y=baseline[g]+effects[D[g]]+rng.normal(0,error_sd,size=sizes[g])
            frame=pd.DataFrame({'Y':y,'S':S[g],'D':D[g],'G_id':g+1,'Ng':sizes[g]})
            if is_cov: frame[['x_1','x_2']]=np.column_stack([
                np.repeat(x1_std[g],sizes[g]),np.repeat(x2_std[g],sizes[g])])
            rows.append(frame)
        return pd.concat(rows,ignore_index=True)
    n=int(n)
    errors=np.column_stack([rng.normal(size=n) for _ in arms])
    w=np.sqrt(20)*(rng.beta(2,2,n)-.5)
    x1=rng.normal(5,2,n); x2=rng.normal(2,1,n)
    baseline=gamma[0]*w+(gamma[1]*x1+gamma[2]*x2 if is_cov else 0)
    potential=baseline[:,None]+effects[None,:]+errors
    order=np.argsort(w)
    if small_strata:
        if n%k: raise ValueError("n must be divisible by k for small-strata designs.")
        S=np.empty(n,int); D=np.empty(n,int); assignment=np.repeat(arms,np.asarray(treat_sizes,int))
        for j,idx in enumerate(order.reshape(-1,k),1): S[idx]=j; D[idx]=rng.permutation(assignment)
    else:
        bounds=np.linspace(-2.25,2.25,int(n_strata)+1)
        S=np.clip(np.digitize(w,bounds[1:-1],right=True)+1,1,int(n_strata))
        probs=np.full((n_strata,len(arms)),1/len(arms)) if allocation_probs is None else np.asarray(allocation_probs,float)
        if probs.shape!=(n_strata,len(arms)) or np.any(probs<=0) or not np.allclose(probs.sum(1),1):
            raise ValueError("allocation_probs must be strictly positive with n_strata rows and len(tau_vec)+1 columns; rows must sum to one.")
        D=np.empty(n,int)
        for s in range(1,n_strata+1):
            idx=np.flatnonzero(S==s)
            if allocation_probs is None:
                active=np.full(len(tau),int(np.floor(len(idx)/(len(tau)+1))))
                control=len(idx)-active.sum()
                D[idx]=rng.permutation(np.repeat(arms,np.r_[control,active]))
            else:
                # Match the R generator: active-arm counts are the floors of
                # their requested probabilities and control receives the
                # remaining observations. Assignment is then randomized
                # within the stratum.
                active=np.floor(probs[s-1,1:]*len(idx)).astype(int)
                control=len(idx)-active.sum()
                assignment=np.repeat(arms,np.r_[control,active])
                D[idx]=rng.permutation(assignment)
    if stratum_effects is not None:
        arr=np.asarray(stratum_effects,float)
        if arr.shape!=(n_strata,) or not np.isfinite(arr).all():
            raise ValueError("stratum_effects must be a finite numeric vector of length n_strata.")
        se=arr[S-1]
    else: se=np.zeros(n)
    if treatment_effects_by_stratum is None: te=effects[D]
    else:
        matrix=np.asarray(treatment_effects_by_stratum,float)
        if matrix.shape!=(n_strata,len(tau)) or not np.isfinite(matrix).all(): raise ValueError("treatment_effects_by_stratum must be finite and have n_strata rows and len(tau_vec) columns.")
        te=np.where(D==0,0,matrix[S-1,np.maximum(D-1,0)])
    y=potential[np.arange(n),D]+se
    if treatment_effects_by_stratum is not None:
        y += te-effects[D]
    out=pd.DataFrame({'Y':y,'S':S,'D':D})
    if is_cov: out[['x_1','x_2']]=np.column_stack([x1,x2])
    return out
