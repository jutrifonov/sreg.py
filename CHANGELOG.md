# Changelog

All notable changes to the Python package are documented here. The 2.1.0
entry mirrors the corresponding R package release.

## 2.1.0

### Estimation and inference

- Corrected large-strata multi-arm variance estimation under individual- and
  cluster-level assignment so other active arms contribute appropriately.
- Corrected small-strata cluster point and variance estimation, including the
  common random denominator and contributions from every treatment arm.
- Corrected mixed cluster inference to use represented-population shares from
  `Ng` and account for random share variation.
- Applied mixed-design covariate adjustment to both small- and large-strata
  components, with targeted diagnostics for unidentified regressions.
- Improved HC1 behavior in degenerate multi-treatment settings.

### Design support and interface

- Added `k` to `sreg()` for uniform small-strata validation and general
  k-tuple mixed designs.
- Improved small/mixed design classification, warnings, and validation under
  individual- and cluster-level assignment.
- Standardized output on the term “large strata.”

### Data generation

- Added mixed-design generation through `mixed_strata` and `n_small`.
- Added `allocation_probs`, `stratum_effects`, and
  `treatment_effects_by_stratum` for customized large-strata individual DGPs.
- Clarified that `n` counts clusters when `cluster=True`, strengthened input
  validation, and made `is_cov=False` omit covariate columns consistently.
- Added `random_state` for reproducible Python simulations.

### Documentation and testing

- Expanded API documentation, examples, plotting documentation, AEJapp
  guidance, and the introductory tutorial for every supported design.
- Added deterministic R 2.1.0 numerical oracles and translations of all 55 R
  `test_that()` blocks.

## 2.0.2

- Maintenance release.

## 2.0.1

- First stable 2.0-series release.

## 2.0.0

- Added small-strata, matched-pair, n-tuple, mixed-design, cluster-design, and
  covariate-adjustment support.
- Added formatted result output and confidence-interval plotting.

## 1.0.1

- Fixed single-covariate adjustment returning an unadjusted result.
- Minor fixes and improvements.

## 1.0.0

- Initial Python release.
