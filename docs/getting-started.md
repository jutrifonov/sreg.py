# A practical guide to sreg

This is the Python counterpart of the R 2.1.0 introductory vignette.

## Overview

`sreg` supports individual- and cluster-level assignment, multiple active
arms, unstratified and large-strata experiments, matched pairs and general
k-tuples, mixed small/large designs, linear adjustment, HC1 standard errors,
simulation, formatted summaries, and confidence-interval plots.

```python
from sreg import AEJapp, sreg, sreg_rgen
```

## Basic workflow and conventions

1. Identify the randomization unit and strata design.
2. Prepare `Y`, `D`, optional `S`, cluster variables, and covariates.
3. Call `sreg()` with options matching the design.
4. Inspect, print, or plot the returned result.

Control must be `D=0`; active treatments must be consecutive integers. Strata
must be consecutive positive integers. Use `S=None` only for unstratified
large-strata inference. For cluster assignment, supply `G_id` and preferably
`Ng`.

| Design | Recommended options |
|---|---|
| No stratification | `S=None, small_strata=False` |
| Large strata | `S=S, small_strata=False` |
| Uniform pairs/k-tuples | `small_strata=True`; `k` is optional validation |
| Mixed pairs/triplets and large strata | `small_strata=True`; automatic detection is available |
| Mixed general k-tuples | `small_strata=True, k=<known size>` |
| Cluster version | additionally supply `G_id` and preferably `Ng` |

## Large strata

```python
large = sreg_rgen(
    n=600, tau_vec=(0.3, 0.7), n_strata=4,
    cluster=False, random_state=101,
)

fit_large_unadjusted = sreg(large.Y, large.S, large.D)
fit_large = sreg(
    large.Y, large.S, large.D,
    X=large[["x_1", "x_2"]],
)
print(fit_large)
```

## Plotting

```python
ax = fit_large.plot(
    treatment_labels=["Program A", "Program B"],
    title="Estimated treatment effects",
    x_axis_title="ATE relative to control",
    bar_fill=("#3B82F6", "#14B8A6"),
    point_fill="white",
)
ax.figure.tight_layout()
```

The return value is a Matplotlib `Axes`, which can be customized or saved with
`ax.figure.savefig("effects.png", dpi=150)`.

## Uniform small strata

```python
small = sreg_rgen(
    n=300, tau_vec=(0.3, 0.7), cluster=False,
    small_strata=True, k=3, treat_sizes=(1, 1, 1),
    random_state=102,
)
fit_small = sreg(
    small.Y, small.S, small.D,
    X=small[["x_1", "x_2"]],
    small_strata=True,
)
```

Here every stratum is a triplet with one unit in each arm. Supplying `k=3` to
`sreg()` is optional in a uniform design but validates the observed size.

## Mixed small and large strata

```python
mixed = sreg_rgen(
    n=120, tau_vec=(0.5,), cluster=False,
    mixed_strata=True, n_small=80, k=4,
    treat_sizes=(2, 2), n_strata=4, random_state=103,
)
fit_mixed = sreg(
    mixed.Y, mixed.S, mixed.D,
    small_strata=True, k=4,
)
```

At least 25% of strata must have the selected small size. The threshold is a
share of strata, not observations. Small and large component estimates are
combined using their represented-population shares. Covariates, when supplied,
are used in both components.

## Cluster-randomized experiments

```python
clustered = sreg_rgen(
    n=60, tau_vec=(0.5,), n_strata=4,
    cluster=True, random_state=104,
)
fit_cluster = sreg(
    clustered.Y, clustered.S, clustered.D,
    G_id=clustered.G_id, Ng=clustered.Ng,
    X=clustered[["x_1", "x_2"]],
)
```

For cluster small/mixed designs, `k` counts distinct clusters. If covariates
vary within a cluster, Python follows R by aggregating them to cluster means
and warning. When `Ng=None`, observed record counts are used and a warning is
issued.

## No stratification

```python
fit_unstratified = sreg(Y, S=None, D=D, X=X, small_strata=False)
```

Do not combine `S=None` with `small_strata=True`.

## AEJapp empirical application

```python
aej = AEJapp()
D = aej["treatment"].replace(3, 0)

aej_unadjusted = sreg(aej["gradesq34"], aej["class_level"], D)
aej_adjusted = sreg(
    aej["gradesq34"], aej["class_level"], D,
    X=aej[["pills_taken", "age_months"]],
)

print(aej_unadjusted["tau_hat"])
print(aej_adjusted["tau_hat"])
```

The R 2.1.0 oracle values are:

- unadjusted estimates `[-0.05112971, 0.40903373]` and standard errors
  `[0.2064541, 0.2065146]`;
- adjusted estimates `[-0.02861589, 0.34608688]` and standard errors
  `[0.1816173, 0.1857249]`.

## HC1

`HC1=True` is the default finite-sample correction. Use `HC1=False` for the
corresponding uncorrected estimator. The choice should follow the prespecified
analysis rather than whichever result gives a preferred standard error.

## Common diagnostics

- A mixed-design warning records that weighted component estimators are used.
- A large-strata warning on recurring small strata indicates that the requested
  variance procedure may not match the randomization design.
- Missing `Ng` causes observed cluster counts to be substituted.
- Insufficient within-cell covariate variation can force unadjusted estimation
  or make the mixed large component unidentified.

## Analysis checklist

Confirm treatment and stratum indexing, the assignment unit, cluster sizes,
the large/small/mixed design classification, fixed allocations in small strata,
the chosen `k` for general mixed designs, and adequate covariate variation.

See [the API reference](api.md), [complete runnable examples](../examples.py),
and [line-by-line examples](../examples_line_by_line.py).

## References

- Bugni, Canay, and Shaikh (2018), “Inference Under Covariate-Adaptive
  Randomization,” *JASA*, https://doi.org/10.1080/01621459.2017.1375934.
- Bugni, Canay, Shaikh, and Tabord-Meehan (2024+), “Inference for Cluster
  Randomized Experiments with Non-ignorable Cluster Sizes,”
  https://doi.org/10.48550/arXiv.2204.08356.
- Jiang, Linton, Tang, and Zhang (2023+), “Improving Estimation Efficiency via
  Regression-Adjustment in Covariate-Adaptive Randomizations with Imperfect
  Compliance,” https://doi.org/10.48550/arXiv.2201.13004.
- Bai et al. (2024), “Covariate adjustment in experiments with matched pairs,”
  *Journal of Econometrics*, https://doi.org/10.1016/j.jeconom.2024.105740.
- Bai (2022), “Optimality of Matched-Pair Designs in Randomized Controlled
  Trials,” *AER*, https://doi.org/10.1257/aer.20201856.
- Bai, Romano, and Shaikh (2022), “Inference in Experiments With Matched
  Pairs,” *JASA*, https://doi.org/10.1080/01621459.2021.1883437.
- Liu (2024+), “Inference for Two-stage Experiments under Covariate-Adaptive
  Randomization,” https://doi.org/10.48550/arXiv.2301.09016.
- Cytrynbaum (2024), “Covariate Adjustment in Stratified Experiments,”
  *Quantitative Economics*, https://doi.org/10.3982/QE2475.
