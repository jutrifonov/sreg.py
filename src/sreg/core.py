r"""
Sreg:Stratified Randomized Experiments
============
The sreg package offers a toolkit for estimating average treatment effects (ATEs) in stratified randomized experiments. 
The package is designed to accommodate scenarios with multiple treatments and cluster-level treatment assignments, 
and accomodates optimal linear covariate adjustment based on baseline observable characteristics. The package 
computes estimators and standard errors based on Bugni, Canay, Shaikh (2018); Bugni, Canay, Shaikh, Tabord-Meehan (2023); 
and Jiang, Linton, Tang, Zhang (2023).

Dependencies: numpy
"""
# The core
import pandas as pd
import numpy as np
import warnings

from .data_check import (check_data_types, check_integers, boolean_check, check_range,
                         check_within_strata_variation,
                         check_within_strata_treatment_variation)
from .check_cluster import check_cluster, check_cluster_lvl
from .result import res_sreg, res_creg
from .small_strata import (classify_strata, res_sreg_small, res_creg_small,
                           combine_mixed, _result)
from .generator import generate
from .output import Sreg



def sreg(Y, S=None, D=None, G_id=None, Ng=None, X=None, HC1=True,
         small_strata=False, k=None):
    """Estimate treatment effects and robust standard errors.

    This is the Python counterpart of R ``sreg()``. Active treatments are
    compared with control (``D=0``). Supplying ``G_id`` selects cluster-level
    assignment; otherwise assignment is individual-level. ``small_strata``
    selects small/mixed inference, while its false value selects the
    large-strata procedure.

    Parameters
    ----------
    Y : array-like
        Numeric observed outcomes, one value per observation.
    S : array-like or None, default=None
        Strata indexed by consecutive positive integers. ``None`` denotes no
        stratification and is incompatible with ``small_strata=True``.
    D : array-like
        Treatments indexed ``0, 1, ...``, where zero is control.
    G_id : array-like or None, default=None
        Cluster identifiers. ``None`` selects individual assignment.
    Ng : array-like or None, default=None
        Represented cluster sizes. If omitted for cluster assignment, observed
        records per cluster are substituted with a warning.
    X : pandas.DataFrame, numpy.ndarray, or None, default=None
        Numeric adjustment covariates. Under cluster assignment, covariates
        that vary within cluster are aggregated to cluster means with a
        warning. In mixed designs they are used in both components.
    HC1 : bool, default=True
        Apply the finite-sample HC1 variance correction.
    small_strata : bool, default=False
        Use the small-strata estimator for a uniform design or the combined
        estimator for a mixed small/large design.
    k : int or None, default=None
        Units per small stratum, or clusters per small stratum under cluster
        assignment. Validates uniform designs and identifies the small
        component in general mixed k-tuple designs.

    Returns
    -------
    Sreg
        Mapping-like result with ``tau_hat``, ``se_rob``, ``t_stat``,
        ``p_value``, ``CI_left``, ``CI_right``, ``data``, ``lin_adj``,
        ``small_strata``, and ``HC1``. Mixed results also contain
        ``mixed_design``, ``res_small``, and ``res_big``; adjustment paths may
        contain ``beta_hat`` or ``ols_iter``.

    Notes
    -----
    In a mixed design, at least 25 percent of strata must have the selected
    small size. If its large component cannot identify requested adjustment
    regressions, reduce the covariate set or use ``X=None``.

    Examples
    --------
    >>> data = sreg_rgen(120, tau_vec=(0.2, 0.5), n_strata=4,
    ...                  cluster=False, random_state=1)
    >>> fit = sreg(data.Y, data.S, data.D,
    ...            X=data[["x_1", "x_2"]])
    >>> fit["tau_hat"].shape
    (2,)
    """

    check_data_types(Y, S, D, G_id, Ng, X)
    check_integers(S, D, G_id, Ng)
    boolean_check(HC1)
    boolean_check(small_strata)
    if k is not None and (isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0):
        raise ValueError("k must be None or a positive integer.")

    if Y is None:
        raise ValueError("Error: Observed outcomes have not been provided (Y = None). Please provide the vector of observed outcomes.")
    if D is None:
        raise ValueError("Error: Treatments have not been provided (D = None). Please provide the vector of treatments.")

    def as_vector(value, name):
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(f"Error: {name} must be a one-dimensional vector or a one-column matrix/data frame.")
        return array

    # R accepts vectors, one-column matrices, data.frames, and tibbles for
    # scalar design variables. Normalize their NumPy/Pandas counterparts.
    Y = as_vector(Y, 'Y')
    S = as_vector(S, 'S')
    D = as_vector(D, 'D')
    G_id = as_vector(G_id, 'G_id')
    Ng = as_vector(Ng, 'Ng')
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
        warnings.warn("The data contains one or more NA (or NaN) values. Proceeding while ignoring these values.",
                      UserWarning, stacklevel=2)
    clean_df = check_df.dropna()

    x_ind = max([i for i, col in enumerate(clean_df.columns) if col in ['D', 'G_id', 'Ng']], default=-1)

    Y = clean_df['Y']
    if S is not None:
        S = clean_df['S'].astype(int)
    D = clean_df['D'].astype(int)
    if G_id is not None:
        G_id = clean_df['G_id'].astype(int)
    if Ng is not None:
        Ng = clean_df['Ng'].astype(int)
    
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

    if small_strata and S is None:
        raise ValueError("Strata indicators are required when small_strata=True.")

    design = pd.DataFrame({'Y': np.asarray(Y), 'S': np.asarray(S) if S is not None else 1,
                           'D': np.asarray(D)})
    if G_id is not None:
        design['G_id'] = np.asarray(G_id)
    if Ng is not None:
        design['Ng'] = np.asarray(Ng)
    elif G_id is not None:
        design['Ng'] = design.groupby('G_id').Y.transform('size')
    if X is not None:
        design = pd.concat([design.reset_index(drop=True), pd.DataFrame(X).reset_index(drop=True)], axis=1)

    if not small_strata and S is not None:
        observed_sizes = (design[['S', 'G_id']].drop_duplicates().groupby('S').size()
                          if G_id is not None else design.groupby('S').size())
        if observed_sizes.nunique() == 1 and observed_sizes.iloc[0] <= 5:
            unit = "clusters" if G_id is not None else "observations"
            warnings.warn(
                f"All strata have the same small number of {unit}, but small_strata=False. "
                "Consider setting small_strata=True to apply estimators designed for such designs.",
                UserWarning, stacklevel=2,
            )
        classify_strata(design, cluster=G_id is not None, small_strata=False, k=k)
    if not small_strata and X is not None:
        x_names=list(pd.DataFrame(X).columns)
        if G_id is None:
            variation=pd.DataFrame(X).reset_index(drop=True)
            variation.columns=x_names; variation.insert(0,'D',np.asarray(D)); variation.insert(0,'S',np.asarray(S) if S is not None else 1)
        else:
            variation=pd.DataFrame(X).reset_index(drop=True); variation.columns=x_names
            variation['G_id']=np.asarray(G_id); variation['S']=np.asarray(S) if S is not None else 1; variation['D']=np.asarray(D)
            variation=variation.groupby('G_id',sort=False).agg({**{c:'mean' for c in x_names},'S':'first','D':'first'}).reset_index()
        if not check_within_strata_variation(variation) or not check_within_strata_treatment_variation(variation):
            warnings.warn("One or more covariates do not vary within one or more stratum-treatment combinations while small_strata=False. Proceeding with the unadjusted estimator.",UserWarning,stacklevel=2)
            X=None

    mixed = False
    classified = design
    if S is not None:
        sizes = (design[['S', 'G_id']].drop_duplicates().groupby('S').size()
                 if G_id is not None else design.groupby('S').size())
        mixed = sizes.nunique() > 1
        if small_strata and not mixed and k is not None and sizes.iloc[0] != k:
            raise ValueError(f"The supplied small-stratum size k = {k} does not match "
                             f"the observed stratum size of {sizes.iloc[0]}.")

    if small_strata and mixed:
        classified = classify_strata(design, cluster=G_id is not None,
                                     small_strata=True, k=k)
        small_mask = classified.stratum_type.eq('small').to_numpy()
        xcols = [c for c in classified.columns
                 if c not in {'Y','S','D','G_id','Ng','stratum_type'}]
        xs = classified.loc[small_mask, xcols] if xcols else None
        xb = classified.loc[~small_mask, xcols] if xcols else None
        if xb is not None:
            large=classified.loc[~small_mask]
            counts=(large[['S','D','G_id']].drop_duplicates().groupby(['S','D']).size()
                    if G_id is not None else large.groupby(['S','D']).size())
            if counts.min() <= len(xcols)+1:
                raise ValueError("The large-strata component of the mixed design cannot support the requested covariate adjustment because treatment-by-stratum regressions are unidentified. Reduce the number of covariates or rerun sreg() with X=None.")
        if G_id is None:
            rs = res_sreg_small(classified.Y[small_mask], classified.S[small_mask],
                                classified.D[small_mask], xs, HC1)
            rb = res_sreg(classified.Y[~small_mask],
                          pd.factorize(classified.S[~small_mask])[0] + 1,
                          classified.D[~small_mask], xb, HC1)
            if xb is not None:
                coeff=np.concatenate([np.ravel(v) for v in rb.get('ols_iter',[])]) if rb.get('ols_iter') else np.array([])
                if coeff.size==0 or not np.isfinite(coeff).all() or not np.isfinite(rb['se_rob']).all():
                    raise ValueError("The large-strata component of the mixed design cannot support the requested covariate adjustment because treatment-by-stratum regressions are unidentified. Reduce the number of covariates or rerun sreg() with X=None.")
            p = small_mask.mean()
            tau, se = combine_mixed(rs, rb, p, p*(1-p), len(classified))
        else:
            rs = res_creg_small(classified.Y[small_mask], classified.S[small_mask],
                                classified.D[small_mask], classified.G_id[small_mask],
                                classified.Ng[small_mask] if 'Ng' in classified else None, xs, HC1)
            rb = res_creg(classified.Y[~small_mask],
                          pd.factorize(classified.S[~small_mask])[0] + 1,
                          classified.D[~small_mask], classified.G_id[~small_mask],
                          classified.Ng[~small_mask] if 'Ng' in classified else None, xb, HC1)
            if xb is not None:
                coeff=np.concatenate([np.ravel(v) for v in rb.get('ols_iter',[])]) if rb.get('ols_iter') else np.array([])
                if coeff.size==0 or not np.isfinite(coeff).all() or not np.isfinite(rb['se_rob']).all():
                    raise ValueError("The large-strata component of the mixed design cannot support the requested covariate adjustment because treatment-by-stratum regressions are unidentified. Reduce the number of covariates or rerun sreg() with X=None.")
            clusters = classified[['G_id','stratum_type','Ng']].drop_duplicates('G_id')
            nsmall = clusters.loc[clusters.stratum_type.eq('small'),'Ng'].sum()
            ntotal = clusters.Ng.sum(); p=nsmall/ntotal
            is_big=clusters.stratum_type.eq('big').astype(float)
            vp=np.mean(clusters.Ng**2*(is_big-(1-p))**2)/(ntotal/len(clusters))**2
            tau,se=combine_mixed(rs,rb,p,vp,len(clusters))
        result = _result(tau, se, classified, beta=rs.get('beta_hat'),
                         lin_adj=rs.get('lin_adj'), ols_iter=rb.get('ols_iter'),
                         res_small=rs, res_big=rb, mixed_design=True)
    elif small_strata and G_id is None:
        result = res_sreg_small(Y, S, D, X, HC1)
    elif small_strata:
        result = res_creg_small(Y, S, D, G_id, Ng, X, HC1)
    elif G_id is None:
        result = res_sreg(Y, S, D, X, HC1)
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
    else:
        check_cluster_lvl(G_id, S, D, Ng)
        result = res_creg(Y, S, D, G_id, Ng, X, HC1)
        if Ng is None:
            warnings.warn("Cluster sizes have not been provided (Ng=None). Ng is assumed to equal the number of available observations in each cluster.",UserWarning,stacklevel=2)
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
        if result['lin_adj'] is not None:
            if not check_cluster(pd.DataFrame({'G_id': result['data']['G_id'], **result['lin_adj']})):
                warnings.warn("sreg cannot use individual-level covariates for covariate adjustment in cluster-randomized experiments. Covariates were aggregated to cluster-level averages.",UserWarning,stacklevel=2)

    result['small_strata'] = small_strata
    result['HC1'] = HC1
    return result if isinstance(result, Sreg) else Sreg(result)

def sreg_rgen(n, Nmax=50, n_strata=10, tau_vec=(0,), gamma_vec=(0.4, 0.2, 1),
              cluster=True, is_cov=True, small_strata=False, k=3,
              treat_sizes=None, mixed_strata=False, n_small=None,
              allocation_probs=None, stratum_effects=None,
              treatment_effects_by_stratum=None, random_state=None):
    """Generate a randomized experiment in a supported design.

    Parameters mirror R ``sreg.rgen()`` using Python ``snake_case`` names.

    Parameters
    ----------
    n : int
        Observations when ``cluster=False``; clusters when ``cluster=True``.
    Nmax : int, default=50
        Maximum generated cluster size.
    n_strata : int, default=10
        Number of large strata.
    tau_vec : sequence of float, default=(0,)
        Active-arm treatment effects relative to control.
    gamma_vec : sequence of float, default=(0.4, 0.2, 1)
        Three DGP coefficients.
    cluster : bool, default=True
        Generate cluster-level assignment.
    is_cov : bool, default=True
        Include generated covariates ``x_1`` and ``x_2``.
    small_strata : bool, default=False
        Generate a uniform small-strata design.
    k : int, default=3
        Units or clusters per small stratum.
    treat_sizes : sequence of int or None, default=None
        Fixed counts for control and each active arm in a small stratum. The
        entries must sum to ``k``.
    mixed_strata : bool, default=False
        Generate both small- and large-strata components.
    n_small : int or None, default=None
        Units or clusters assigned to the small component; divisible by ``k``.
    allocation_probs : array-like or None, default=None
        Optional common or stratum-specific large-strata allocations.
    stratum_effects : array-like or None, default=None
        Optional outcome intercept by stratum.
    treatment_effects_by_stratum : array-like or None, default=None
        Optional treatment effects by stratum and active arm.
    random_state : int, numpy.random.Generator, or None, default=None
        Python random-state input. Equal R and Python seeds do not imply equal
        samples because their random-number generators differ.

    Returns
    -------
    pandas.DataFrame
        ``Y``, ``S``, and ``D`` plus ``G_id`` and ``Ng`` for cluster designs
        and ``x_1``, ``x_2`` when ``is_cov=True``.

    Examples
    --------
    >>> data = sreg_rgen(60, tau_vec=(0.2, 0.5), cluster=False,
    ...                  small_strata=True, k=3,
    ...                  treat_sizes=(1, 1, 1), random_state=2)
    >>> set(["Y", "S", "D"]).issubset(data.columns)
    True
    """
    return generate(n, Nmax, n_strata, tau_vec, gamma_vec, cluster, is_cov,
                    small_strata, k, treat_sizes, mixed_strata, n_small,
                    allocation_probs, stratum_effects,
                    treatment_effects_by_stratum, random_state)

import pkgutil
import io

def AEJapp():
    """Return the bundled AEJ application data.

    Returns
    -------
    pandas.DataFrame
        A fresh data frame with 215 observations and 62 variables from Chong
        et al. (2016), *Iron Deficiency and Schooling Attainment in Peru*.

    Examples
    --------
    >>> data = AEJapp()
    >>> data.shape
    (215, 62)
    """
    data = pkgutil.get_data('sreg', 'data/AEJapp.csv')
    return pd.read_csv(io.BytesIO(data))
