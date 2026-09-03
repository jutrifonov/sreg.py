# sreg: Stratified Randomized Experiments (Python)

<img src="https://github.com/jutrifonov/sreg.dev/blob/main/logo.png" align="right" height="220" alt="sreg logo" />

`sreg` estimates average treatment effects and robust standard errors in
stratified randomized experiments. This pure-Python implementation follows
the estimators and behavior of the R `sreg` package version 2.1.0.

It supports:

- individual- and cluster-level treatment assignment;
- one or multiple active treatments relative to control;
- unstratified and large-strata designs;
- matched pairs and general k-tuples;
- mixed small- and large-strata designs;
- unadjusted and optimally linearly adjusted estimation;
- HC1 finite-sample corrections;
- simulated data generation; and
- formatted results and confidence-interval plots.

## Installation

From PyPI after release:

```bash
python -m pip install sreg
```

From GitHub:

```bash
python -m pip install "git+https://github.com/jutrifonov/sreg_py.git"
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Basic use

```python
from sreg import sreg, sreg_rgen

data = sreg_rgen(
    n=600,
    tau_vec=(0.2, 0.8),
    n_strata=5,
    cluster=False,
    is_cov=True,
    random_state=101,
)

fit = sreg(
    Y=data["Y"],
    S=data["S"],
    D=data["D"],
    X=data[["x_1", "x_2"]],
)

print(fit)
print(fit["tau_hat"])
print(fit["se_rob"])
fit.plot()
```

Treatment must be indexed `0, 1, 2, ...`, with `0` as control. Strata must be
indexed `1, 2, ...`. Set `S=None` for an unstratified experiment.

## Main estimator

```python
sreg(
    Y, S=None, D=None,
    G_id=None, Ng=None,
    X=None, HC1=True,
    small_strata=False, k=None,
)
```

The signature mirrors R, with idiomatic Python names:

| R | Python |
|---|---|
| `G.id` | `G_id` |
| `small.strata` | `small_strata` |
| `NULL`, `TRUE`, `FALSE` | `None`, `True`, `False` |
| `fit$tau.hat` | `fit["tau_hat"]` |
| `fit$se.rob` | `fit["se_rob"]` |

`G_id=None` selects individual assignment. Supplying `G_id` selects cluster
assignment, where `Ng` is the represented cluster size. If `Ng` is omitted,
observed records per cluster are substituted with a warning.

`small_strata=False` selects large-strata inference. With
`small_strata=True`, a uniform size selects small-strata inference and varying
sizes select the mixed estimator. `k` validates uniform designs and identifies
the small component in general mixed k-tuple designs.

See the [complete API reference](https://github.com/jutrifonov/sreg_py/blob/main/docs/api.md) for every argument, result field,
warning, and plotting option.

## Small strata

```python
small = sreg_rgen(
    n=300,
    tau_vec=(0.2, 0.8),
    cluster=False,
    small_strata=True,
    k=3,
    treat_sizes=(1, 1, 1),
    random_state=201,
)

fit_small = sreg(
    small.Y, small.S, small.D,
    X=small[["x_1", "x_2"]],
    small_strata=True,
    k=3,
)
```

## Mixed strata

```python
mixed = sreg_rgen(
    n=120,
    tau_vec=(0.5,),
    cluster=False,
    mixed_strata=True,
    n_small=80,
    k=4,
    treat_sizes=(2, 2),
    n_strata=4,
    random_state=202,
)

fit_mixed = sreg(
    mixed.Y, mixed.S, mixed.D,
    small_strata=True,
    k=4,
)
```

## Cluster assignment

```python
clustered = sreg_rgen(
    n=60,
    tau_vec=(0.5,),
    n_strata=4,
    cluster=True,
    random_state=203,
)

fit_cluster = sreg(
    clustered.Y, clustered.S, clustered.D,
    G_id=clustered.G_id,
    Ng=clustered.Ng,
    X=clustered[["x_1", "x_2"]],
)
```

For cluster-randomized small or mixed designs, `k` counts clusters per small
stratum. Individual-varying covariates are aggregated to cluster means with a
warning, matching the R package.

## AEJapp application

```python
from sreg import AEJapp

aej = AEJapp()
D = aej["treatment"].replace(3, 0)

unadjusted = sreg(aej["gradesq34"], aej["class_level"], D)
adjusted = sreg(
    aej["gradesq34"], aej["class_level"], D,
    X=aej[["pills_taken", "age_months"]],
)
```

R 2.1.0 and Python return the same oracle results:

| Specification | `tau_hat` | `se_rob` |
|---|---|---|
| Unadjusted | `[-0.05112971, 0.40903373]` | `[0.2064541, 0.2065146]` |
| Adjusted | `[-0.02861589, 0.34608688]` | `[0.1816173, 0.1857249]` |

## Plotting

```python
ax = adjusted.plot(
    treatment_labels=["Treatment 1", "Treatment 2"],
    title="AEJapp treatment effects",
    x_axis_title="ATE relative to control",
    bar_fill=("#3B82F6", "#14B8A6"),
)
ax.figure.savefig("effects.png", dpi=150, bbox_inches="tight")
```

The Python method returns a Matplotlib `Axes`; R's method returns a `ggplot`.

## Data generation

```python
sreg_rgen(
    n, Nmax=50, n_strata=10,
    tau_vec=(0,), gamma_vec=(0.4, 0.2, 1),
    cluster=True, is_cov=True,
    small_strata=False, k=3, treat_sizes=None,
    mixed_strata=False, n_small=None,
    allocation_probs=None, stratum_effects=None,
    treatment_effects_by_stratum=None,
    random_state=None,
)
```

When `cluster=False`, `n` counts observations. When `cluster=True`, `n` counts
clusters. R and NumPy use different pseudorandom streams, so the same numeric
seed does not create identical simulated samples across languages; estimation
on identical input data is numerically equivalent.

## Documentation and examples

- [Practical guide—the Python version of the R vignette](https://github.com/jutrifonov/sreg_py/blob/main/docs/getting-started.md)
- [Full API reference](https://github.com/jutrifonov/sreg_py/blob/main/docs/api.md)
- [Complete runnable examples](https://github.com/jutrifonov/sreg_py/blob/main/examples.py)
- [VS Code line-by-line examples](https://github.com/jutrifonov/sreg_py/blob/main/examples_line_by_line.py)
- [Release history](https://github.com/jutrifonov/sreg_py/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/jutrifonov/sreg_py/blob/main/CITATION.cff)
- [R-to-pytest translation checklist](https://github.com/jutrifonov/sreg_py/blob/main/tests/R_PARITY_CHECKLIST.md)

Run all tests with:

```bash
python -m pytest -q
```

The current suite contains 145 collected tests and translations of all 55 R
2.1.0 `test_that()` blocks, plus deterministic cross-language oracle tests.

## Authors

- Juri Trifonov
- Yuehao Bai
- Azeem Shaikh
- Max Tabord-Meehan

## License

MIT. See [LICENSE](https://github.com/jutrifonov/sreg_py/blob/main/LICENSE).

## References

The methodological references include Bugni, Canay, and Shaikh (2018); Bugni,
Canay, Shaikh, and Tabord-Meehan (2024+); Jiang, Linton, Tang, and Zhang
(2023+); Bai et al. (2024); Bai (2022); Bai, Romano, and Shaikh (2022); Liu
(2024+); and Cytrynbaum (2024). Full citations and DOI links are provided in
the [practical guide](https://github.com/jutrifonov/sreg_py/blob/main/docs/getting-started.md).
