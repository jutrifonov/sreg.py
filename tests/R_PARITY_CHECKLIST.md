# R 2.1.0 test translation checklist

This file tracks every `test_that()` block in the R 2.1.0 package. A checked
item means that its behavior and expectations have a pytest counterpart; it
does not mean that R and Python must produce identical pseudorandom draws.

## Specialized test files (30/30 translated)

- [x] cluster large-strata variance includes clusters in other treatment arms
- [x] mixed cluster weights use supplied cluster population sizes
- [x] mixed cluster weights infer cluster sizes from available observations
- [x] mixed cluster adjustment is applied to both components
- [x] mixed cluster variance uses cluster-level component-share variation
- [x] small-strata cluster point estimator uses expanded outcomes and a common denominator
- [x] small-strata cluster inference supports multiple arms
- [x] binary small-strata variance is the multi-arm formula with two arms
- [x] small-strata cluster adjustment works when cluster sizes are inferred
- [x] mixed-design warning reports the detected individual-level k
- [x] mixed-design warning detects k from cluster counts
- [x] fewer than 25 percent at one size does not produce a mixed warning
- [x] sreg estimates individual-level mixed designs with 4-tuples
- [x] sreg estimates cluster-level mixed designs with 4-tuples
- [x] general k must be supplied when automatic detection cannot identify it
- [x] explicit k works for uniform k-tuple designs and is validated
- [x] individual mixed adjustment is applied to both components
- [x] individual mixed adjustment reports unidentified large regressions
- [x] cluster mixed adjustment reports unidentified large regressions
- [x] unadjusted mixed estimation remains available after adjustment failure
- [x] is.cov controls covariate columns in large-strata cluster designs
- [x] large-strata custom DGP defaults preserve generated data
- [x] large-strata generator accepts stratum-specific allocations
- [x] large-strata generator applies custom outcome effects
- [x] custom large-strata arguments are validated
- [x] sreg.rgen generates mixed individual-level designs
- [x] sreg.rgen generates mixed cluster-level designs
- [x] mixed sreg.rgen validates its component sizes
- [x] existing sreg.rgen calls retain their behavior
- [x] mixed sreg.rgen derives an allocation when treat.sizes is omitted

## test-core.R (25/25 translated)

- [x] simulations without clusters work
- [x] simulations with clusters work
- [x] One or more covariates do not vary within one or more stratum-treatment combinations while small.strata = FALSE
- [x] individual level X warning works
- [x] no cluster sizes warning works
- [x] data contains one or more NA (or NaN) values warning works
- [x] skipped values in range of S/D works
- [x] non cluster-level error for S, D, Ng works
- [x] empirical example works
- [x] dgp.po warning work
- [x] individual data: small strata, option: small strata
- [x] individual data: large strata, option: large strata
- [x] individual data: small strata, option: large strata
- [x] individual data: large strata, option: small strata
- [x] individual data: mixed design, option: small strata
- [x] individual data: mixed design, option: large strata
- [x] cluster data: small strata, option: small strata
- [x] cluster data: large strata, option: large strata
- [x] cluster data: small strata, option: large strata
- [x] cluster data: large strata, option: small strata
- [x] cluster data: mixed design, option: small strata
- [x] cluster data: mixed design, option: large strata
- [x] print.sreg outputs expected information for large strata
- [x] print.sreg outputs expected information for small strata
- [x] plot.sreg works and returns a plot object
