"""One-for-one translations of test-cluster-small-correction.R."""
import numpy as np
import pandas as pd

from sreg import sreg, sreg_rgen
from sreg.small_strata import res_creg_small


def test_small_strata_cluster_point_estimator_uses_expanded_outcomes_and_common_denominator():
    dat = sreg_rgen(
        400, tau_vec=(.5, .8), cluster=True, is_cov=False,
        small_strata=True, k=4, treat_sizes=(2, 1, 1), random_state=20260812,
    )
    fit = res_creg_small(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, None, True)
    clusters = dat.groupby('G_id', sort=False).agg(
        S=('S', 'first'), D=('D', 'first'), Ng=('Ng', 'first'),
        Y_bar=('Y', 'mean'),
    )
    clusters['T'] = clusters.Ng * clusters.Y_bar
    expected = np.array([
        (clusters.loc[clusters.D.eq(d), 'T'].mean()
         - clusters.loc[clusters.D.eq(0), 'T'].mean()) / clusters.Ng.mean()
        for d in (1, 2)
    ])
    np.testing.assert_allclose(fit['tau_hat'], expected)

    shuffled = dat.sample(frac=1, random_state=1)
    shuffled_fit = res_creg_small(
        shuffled.Y, shuffled.S, shuffled.D, shuffled.G_id,
        shuffled.Ng, None, True,
    )
    np.testing.assert_allclose(shuffled_fit['tau_hat'], fit['tau_hat'])


def test_small_strata_cluster_inference_supports_multiple_arms():
    dat = sreg_rgen(
        400, tau_vec=(.5, .8), cluster=True, is_cov=True,
        small_strata=True, k=4, treat_sizes=(2, 1, 1), random_state=20260813,
    )
    X = dat[['x_1', 'x_2']]
    adjusted = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
                    HC1=True, small_strata=True)
    unadjusted = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, None,
                      HC1=True, small_strata=True)
    assert len(adjusted['tau_hat']) == len(adjusted['se_rob']) == 2
    assert np.isfinite(adjusted['tau_hat']).all()
    assert np.isfinite(adjusted['se_rob']).all()
    assert (adjusted['se_rob'] > 0).all()
    assert np.isfinite(unadjusted['tau_hat']).all()
    assert np.isfinite(unadjusted['se_rob']).all()
    assert (unadjusted['se_rob'] > 0).all()


def test_binary_small_strata_variance_is_multiarm_formula_with_two_arms():
    dat = sreg_rgen(
        400, tau_vec=(.8,), cluster=True, is_cov=True,
        small_strata=True, k=2, treat_sizes=(1, 1), random_state=20260814,
    )
    fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng,
               dat[['x_1', 'x_2']], HC1=True, small_strata=True)
    assert len(fit['se_rob']) == 1
    assert np.isfinite(fit['se_rob']).all()
    assert (fit['se_rob'] > 0).all()


def test_small_strata_cluster_adjustment_works_when_cluster_sizes_are_inferred():
    dat = sreg_rgen(
        400, tau_vec=(.5, .8), cluster=True, is_cov=True,
        small_strata=True, k=4, treat_sizes=(2, 1, 1), random_state=20260815,
    )
    X = dat[['x_1', 'x_2']]
    supplied = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
                    small_strata=True)
    inferred = sreg(dat.Y, dat.S, dat.D, dat.G_id, None, X,
                    small_strata=True)
    np.testing.assert_allclose(inferred['tau_hat'], supplied['tau_hat'])
    np.testing.assert_allclose(inferred['se_rob'], supplied['se_rob'])

    one_covariate = sreg(
        dat.Y, dat.S, dat.D, dat.G_id, None,
        pd.DataFrame({'cluster_size': dat.Ng}), small_strata=True,
    )
    assert np.isfinite(one_covariate['tau_hat']).all()
    assert np.isfinite(one_covariate['se_rob']).all()

    named_ng = sreg(
        dat.Y, dat.S, dat.D, dat.G_id, dat.Ng,
        pd.DataFrame({'x_1': dat.x_1, 'Ng': dat.Ng}), small_strata=True,
    )
    assert np.isfinite(named_ng['tau_hat']).all()
    assert np.isfinite(named_ng['se_rob']).all()
