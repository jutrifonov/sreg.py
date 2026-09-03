# API reference

The Python API follows the R 2.1.0 API, using Python's `snake_case` naming and
`None` in place of R's dotted names and `NULL`.

## Name correspondence

| R | Python |
|---|---|
| `sreg()` | `sreg()` |
| `sreg.rgen()` | `sreg_rgen()` |
| `G.id` | `G_id` |
| `small.strata` | `small_strata` |
| `tau.vec` | `tau_vec` |
| `gamma.vec` | `gamma_vec` |
| `treat.sizes` | `treat_sizes` |
| `mixed.strata` | `mixed_strata` |
| `n.small` | `n_small` |
| `allocation.probs` | `allocation_probs` |
| `stratum.effects` | `stratum_effects` |
| `treatment.effects.by.stratum` | `treatment_effects_by_stratum` |
| `NULL`, `TRUE`, `FALSE` | `None`, `True`, `False` |

## `sreg`

```python
sreg(
    Y, S=None, D=None, G_id=None, Ng=None, X=None, HC1=True,
    small_strata=False, k=None,
)
```

Estimates one ATE for every active treatment relative to control and its
robust standard error.

### Parameters

- `Y`: one-dimensional numeric outcome data.
- `S`: consecutive positive stratum indicators (`1, 2, ...`), or `None` for
  an unstratified large-strata analysis.
- `D`: consecutive treatment indicators (`0, 1, ...`), with `0` as control.
- `G_id`: cluster identifiers, or `None` for individual assignment.
- `Ng`: represented cluster sizes. When omitted for a cluster design, observed
  records per cluster are used and a warning is issued.
- `X`: numeric `pandas.DataFrame` or two-dimensional NumPy array of adjustment
  covariates, or `None` for unadjusted estimation. Individual-varying cluster
  covariates are aggregated to cluster means with a warning.
- `HC1`: whether to apply the finite-sample HC1 variance correction.
- `small_strata`: selects small/mixed-strata inference when true; false selects
  large-strata inference.
- `k`: optional positive small-stratum size. Under cluster assignment this is
  the number of clusters, not individual records. It validates uniform designs
  and identifies the small component in general mixed k-tuple designs.

### Returns

An `Sreg` mapping-like result containing:

| Key | Meaning |
|---|---|
| `tau_hat` | ATE estimates relative to control |
| `se_rob` | Robust standard errors |
| `t_stat` | Test statistics |
| `p_value` | Two-sided normal p-values |
| `CI_left`, `CI_right` | 95% confidence bounds |
| `as_CI` | Combined confidence bounds, when produced by the estimator path |
| `data` | Analysis data retained by the fit |
| `lin_adj` | Covariates used for adjustment, or `None` |
| `small_strata`, `HC1` | Recorded analysis options |
| `mixed_design` | True for a combined small/large estimator |
| `res_small`, `res_big` | Component results for a mixed design |
| `beta_hat`, `ols_iter` | Adjustment coefficients where applicable |

Results support `fit["tau_hat"]`, `fit.get(...)`, `print(fit)`, and
`fit.plot(...)`.

## `sreg_rgen`

```python
sreg_rgen(
    n, Nmax=50, n_strata=10, tau_vec=(0,), gamma_vec=(0.4, 0.2, 1),
    cluster=True, is_cov=True, small_strata=False, k=3,
    treat_sizes=None, mixed_strata=False, n_small=None,
    allocation_probs=None, stratum_effects=None,
    treatment_effects_by_stratum=None, random_state=None,
)
```

Generates data representing the supported randomized designs. `n` is the
number of observations for individual assignment and the number of clusters
for cluster assignment.

- `Nmax`: maximum generated cluster size.
- `n_strata`: number of large strata.
- `tau_vec`: active-arm effects relative to control; its length determines the
  number of active arms.
- `gamma_vec`: three outcome/covariate DGP coefficients.
- `cluster`: generate cluster rather than individual assignment.
- `is_cov`: include `x_1` and `x_2` in the returned data.
- `small_strata`: generate a uniform small-strata design.
- `k`: units or clusters per small stratum.
- `treat_sizes`: fixed control/active-arm counts within each small stratum;
  entries must sum to `k`.
- `mixed_strata`, `n_small`: generate small and large components and specify
  the number of units/clusters in the small component.
- `allocation_probs`: optional common or stratum-specific large-strata
  allocation probabilities.
- `stratum_effects`: optional stratum intercept effects.
- `treatment_effects_by_stratum`: optional stratum-by-arm treatment effects.
- `random_state`: NumPy seed or generator input for reproducibility. R and
  NumPy use different random streams, so equal numeric seeds do not imply
  identical simulated samples across languages.

Returns a `pandas.DataFrame` with `Y`, `S`, `D`; cluster designs additionally
contain `G_id`, `Ng`; and `is_cov=True` additionally returns `x_1`, `x_2`.

## `AEJapp`

```python
AEJapp()
```

Returns a fresh `pandas.DataFrame` containing 215 observations and 62 columns
from Chong et al. (2016). See the practical guide for the replicated analysis.

## `Sreg.plot`

```python
fit.plot(
    level=0.95, ax=None, treatment_labels=None,
    title="Estimated ATEs with Confidence Intervals", bar_fill=None,
    point_shape="D", point_size=3, point_fill="white", point_stroke=1.2,
    point_color="black", label_color="black", label_size=4,
    bg_color=None, grid=True, zero_line=True,
    y_axis_title=None, x_axis_title=None, **kwargs,
)
```

Plots normal confidence intervals and returns a Matplotlib `Axes`. Most option
names mirror R's `plot.sreg`; `level` and `ax` are Python additions. Integer R
shape codes 21–25 and Matplotlib marker strings are accepted.

## Printing

`print(fit)` is the Python equivalent of R's `print.sreg(fit)`. It reports the
design, assignment level, HC1 choice, covariates, estimates, standard errors,
test statistics, p-values, confidence intervals, and significance codes.
