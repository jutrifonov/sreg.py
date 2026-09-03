"""Block-by-block translations of tests/testthat/test-core.R."""
import numpy as np
import pandas as pd
import pytest

from sreg import sreg, sreg_rgen
from sreg.dgp_po import dgp_po_sreg


def test_simulations_without_clusters_work():
    """Translation of the first test-core.R block.

    R's seed-specific estimates are covered separately by deterministic oracle
    fixtures because R and NumPy do not share a random-number stream.
    """
    sim = sreg_rgen(
        1000, tau_vec=(.2, .5), n_strata=10, cluster=False,
        is_cov=True, random_state=123,
    )
    Y, S, D = (sim[c].to_numpy() for c in ('Y', 'S', 'D'))
    X = sim[['x_1', 'x_2']]

    for fit in (
        sreg(Y, S, D, X=X),
        sreg(Y, S, D, X=None),
        sreg(Y, None, D, X=X),
        sreg(Y, None, D, X=None),
    ):
        assert len(fit['tau_hat']) == len(fit['se_rob']) == 2
        assert np.isfinite(fit['tau_hat']).all()
        assert np.isfinite(fit['se_rob']).all()
        assert (fit['se_rob'] > 0).all()

    for bad_hc1 in (5, 'TRUE'):
        with pytest.raises(ValueError, match='either True or False'):
            sreg(Y, S, D, X=X, HC1=bad_hc1)
    with pytest.raises(ValueError, match='different type'):
        sreg(Y.tolist(), S, D, X=X)

    bad_s = S.astype(float).copy(); bad_s[1] = 2.5
    with pytest.raises(ValueError, match='integer values'):
        sreg(Y, bad_s, D)
    bad_d = D.astype(float).copy(); bad_d[4] = .5
    with pytest.raises(ValueError, match='integer values'):
        sreg(Y, S, bad_d, X=X)

    nan_y = Y.copy(); nan_y[:10] = np.nan
    nan_x = X.copy(); nan_x.iloc[9:12] = np.nan
    with pytest.warns(UserWarning, match='ignoring these values'):
        fit = sreg(nan_y, S, D, X=nan_x)
    assert np.isfinite(fit['tau_hat']).all()

    nan_s = S.astype(float).copy(); nan_s[11] = np.nan
    with pytest.warns(UserWarning, match='ignoring these values'):
        sreg(Y, nan_s, D, X=X)

    bad_s = S.copy(); bad_s[0] = 0
    with pytest.raises(ValueError, match='strata should be indexed'):
        sreg(Y, bad_s, D, X=X)
    bad_s[9] = -1
    with pytest.raises(ValueError, match='strata should be indexed'):
        sreg(Y, bad_s, D, X=X)

    bad_d = D.copy(); bad_d[:3] = -1
    with pytest.raises(ValueError, match='treatments should be indexed'):
        sreg(Y, S, bad_d, X=X)
    with pytest.raises(ValueError, match='treatments should be indexed'):
        sreg(Y, None, bad_d, X=None)


def test_simulations_with_clusters_work():
    """Translation of the second test-core.R block."""
    sim = sreg_rgen(
        100, tau_vec=(.2, .8), n_strata=4, cluster=True,
        Nmax=50, is_cov=True, random_state=123,
    )
    Y, S, D, G, Ng = (sim[c].to_numpy() for c in ('Y', 'S', 'D', 'G_id', 'Ng'))
    X = sim[['x_1', 'x_2']]

    fits = (
        sreg(Y, S, D, G, Ng, X),
        sreg(Y, S, D, G, Ng, None),
        sreg(Y, None, D, G, Ng, X),
        sreg(Y, None, D, G, Ng, None),
        sreg(Y, S, D, None, None, X),
        sreg(Y, S, D, G, Ng, pd.DataFrame({'Ng': Ng, 'x_1': sim.x_1, 'x_2': sim.x_2})),
    )
    for fit in fits:
        assert len(fit['tau_hat']) == len(fit['se_rob']) == 2
        assert np.isfinite(fit['tau_hat']).all()
        assert np.isfinite(fit['se_rob']).all()

    for bad_hc1 in (5, 'TRUE'):
        with pytest.raises(ValueError, match='either True or False'):
            sreg(Y, S, D, G, Ng, X, HC1=bad_hc1)
    with pytest.raises(ValueError, match='Observed outcomes'):
        sreg(None, S, D, G, Ng, X)
    with pytest.raises(ValueError, match='Treatments'):
        sreg(Y, S, None, G, Ng, X)

    bad_inputs = (
        dict(S=S.astype(str)), dict(Y=Y.astype(str)), dict(D=D.astype(str)),
        dict(G_id=G.tolist()), dict(Ng=Ng.tolist()), dict(X=X.astype(str)),
    )
    base = dict(Y=Y, S=S, D=D, G_id=G, Ng=Ng, X=X)
    for replacement in bad_inputs:
        args = {**base, **replacement}
        with pytest.raises(ValueError, match='different type'):
            sreg(**args)

    # Python equivalents of R's accepted one-column matrix/data-frame inputs.
    sreg(Y[:, None], S[:, None], D[:, None], G[:, None], Ng[:, None], X)
    sreg(pd.DataFrame({'Y': Y}), pd.DataFrame({'S': S}),
         pd.DataFrame({'D': D}), pd.DataFrame({'G': G}),
         pd.DataFrame({'Ng': Ng}), X)

    for field, values in (('S', S), ('D', D), ('G_id', G), ('Ng', Ng)):
        bad = values.astype(float).copy(); bad[1] += .5
        args = dict(base); args[field] = bad
        with pytest.raises(ValueError, match='integer values'):
            sreg(**args)

    bad_s = S.copy(); bad_s[0] = 0
    with pytest.raises(ValueError, match='strata should be indexed'):
        sreg(Y, bad_s, D, G, Ng, X)
    bad_d = D.copy(); bad_d[:40] = -1
    with pytest.raises(ValueError, match='treatments should be indexed'):
        sreg(Y, S, bad_d, G, Ng, X)


def test_covariates_without_stratum_treatment_variation_warn_and_fall_back():
    n = 72
    S = np.repeat(np.arange(1, 4), 24)
    D = np.tile(np.repeat(np.arange(3), 8), 3)
    Y = np.linspace(-1, 2, n) + .3 * D
    X = pd.DataFrame({'constant_in_cell': S * 10 + D, 'varying': np.arange(n)})
    with pytest.warns(UserWarning, match='do not vary within one or more stratum-treatment'):
        fit = sreg(Y, S, D, X=X)
    assert fit['lin_adj'] is None

    dat = sreg_rgen(90, tau_vec=(.2, .8), n_strata=10, cluster=True,
                    random_state=123)
    cluster_x = dat[['G_id', 'S', 'D']].drop_duplicates('G_id')
    cell_value = {(r.G_id): r.S * 10 + r.D for r in cluster_x.itertuples()}
    Xc = pd.DataFrame({'constant_in_cell': dat.G_id.map(cell_value)})
    with pytest.warns(UserWarning, match='do not vary within one or more stratum-treatment'):
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, Xc)
    assert fit['lin_adj'] is None


def test_individual_level_covariate_warning_works():
    dat = sreg_rgen(100, tau_vec=(.2, .8), n_strata=4, cluster=True,
                    random_state=123)
    X = dat[['x_1', 'x_2']].copy()
    X.iloc[0, 0] += 1
    with pytest.warns(UserWarning, match='cannot use individual-level covariates'):
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X)
    assert np.isfinite(fit['tau_hat']).all()


def test_no_cluster_sizes_warning_works():
    dat = sreg_rgen(100, tau_vec=(.2, .8), n_strata=4, cluster=True,
                    random_state=123)
    with pytest.warns(UserWarning, match='Cluster sizes have not been provided'):
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, None, dat[['x_1', 'x_2']])
    assert np.isfinite(fit['tau_hat']).all()


def test_missing_values_warning_works_for_every_input_location():
    dat = sreg_rgen(100, tau_vec=(.2, .8), n_strata=4, cluster=True,
                    random_state=123)
    base = dict(Y=dat.Y.to_numpy(), S=dat.S.to_numpy(), D=dat.D.to_numpy(),
                G_id=dat.G_id.to_numpy(), Ng=dat.Ng.to_numpy(),
                X=dat[['x_1', 'x_2']])
    cases = []
    for field in ('Y', 'S', 'D', 'G_id', 'Ng'):
        args = {**base}; values = np.asarray(args[field], dtype=float).copy()
        values[0] = np.nan; args[field] = values; cases.append(args)
    args = {**base}; args['X'] = base['X'].copy(); args['X'].iloc[0, 0] = np.nan
    cases.append(args)
    for args in cases:
        with pytest.warns(UserWarning, match='ignoring these values'):
            fit = sreg(**args)
        assert np.isfinite(fit['tau_hat']).all()


def test_skipped_values_in_range_of_strata_and_treatments_are_rejected():
    dat = sreg_rgen(100, tau_vec=(.2, .5), n_strata=5, cluster=False,
                    random_state=123)
    bad_s = dat.S.to_numpy().copy(); bad_s[bad_s == 4] = 1
    with pytest.raises(ValueError, match='skipped values in the range'):
        sreg(dat.Y.to_numpy(), bad_s, dat.D.to_numpy(), X=dat[['x_1', 'x_2']])
    bad_d = dat.D.to_numpy().copy(); bad_d[bad_d == 1] = 2
    with pytest.raises(ValueError, match='skipped values in the range'):
        sreg(dat.Y.to_numpy(), dat.S.to_numpy(), bad_d, X=dat[['x_1', 'x_2']])


def test_strata_treatment_and_size_must_be_cluster_level():
    dat = sreg_rgen(100, tau_vec=(.2, .5), n_strata=5, cluster=True,
                    random_state=123)
    base = dict(Y=dat.Y.to_numpy(), S=dat.S.to_numpy(), D=dat.D.to_numpy(),
                G_id=dat.G_id.to_numpy(), Ng=dat.Ng.to_numpy(),
                X=dat[['x_1', 'x_2']])
    first_cluster = np.flatnonzero(base['G_id'] == base['G_id'][0])
    for field, delta in (('S', 1), ('D', 1), ('Ng', 10)):
        args = {**base}; values = args[field].copy()
        values[first_cluster[0]] += delta; args[field] = values
        with pytest.raises(ValueError, match='must be consistent within each cluster'):
            sreg(**args)


def test_dgp_po_reports_treatment_count_mismatch():
    with pytest.raises(ValueError, match="number of treatments doesn't match"):
        dgp_po_sreg(100, theta_vec=(0, .5), n_treat=3,
                    gamma_vec=(.4, .2, 1))


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('adjustment', [None, ('x_1',), ('x_1', 'x_2')])
def test_individual_small_data_with_small_option(hc1, adjustment):
    dat = sreg_rgen(300, tau_vec=(.2, .8), cluster=False,
                    small_strata=True, k=3, treat_sizes=(1, 1, 1),
                    random_state=123)
    X = None if adjustment is None else dat[list(adjustment)]
    fit = sreg(dat.Y, dat.S, dat.D, X=X, HC1=hc1, small_strata=True)
    assert fit['small_strata'] is True
    assert fit.get('mixed_design', False) is False
    assert len(fit['tau_hat']) == 2
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()
    with pytest.raises(ValueError, match='Strata indicators are required'):
        sreg(dat.Y, None, dat.D, X=X, HC1=hc1, small_strata=True)


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('adjustment', [None, ('x_1',), ('x_1', 'x_2')])
def test_individual_large_data_with_large_option(hc1, adjustment):
    dat = sreg_rgen(1000, tau_vec=(.2, .9, 1.5), n_strata=4,
                    cluster=False, random_state=123)
    X = None if adjustment is None else dat[list(adjustment)]
    fit = sreg(dat.Y, dat.S, dat.D, X=X, HC1=hc1, small_strata=False)
    assert fit['small_strata'] is False
    assert len(fit['tau_hat']) == 3
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('hc1', [True, False])
def test_individual_small_data_with_large_option_warns(hc1):
    dat = sreg_rgen(900, tau_vec=(.2, .8), cluster=False,
                    small_strata=True, k=3, treat_sizes=(1, 1, 1),
                    random_state=123)
    with pytest.warns(UserWarning) as record:
        fit = sreg(dat.Y, dat.S, dat.D, X=dat[['x_1', 'x_2']],
                   HC1=hc1, small_strata=False)
    messages = [str(item.message) for item in record]
    assert any('same small number of observations' in message for message in messages)
    assert any('At least 25% of strata are small' in message for message in messages)
    assert fit['lin_adj'] is None
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('with_covariates', [True, False])
def test_individual_large_data_with_small_option_is_rejected(with_covariates):
    dat = sreg_rgen(1000, tau_vec=(.2, .9), n_strata=20,
                    cluster=False, random_state=123)
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.raises(ValueError, match='either all strata are large|too few strata'):
        sreg(dat.Y, dat.S, dat.D, X=X, small_strata=True)
    with pytest.raises(ValueError, match='Strata indicators are required'):
        sreg(dat.Y, None, dat.D, X=X, small_strata=True)


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('with_covariates', [True, False])
def test_individual_mixed_data_with_small_option(hc1, with_covariates):
    dat = sreg_rgen(
        600, tau_vec=(.2, .9), n_strata=4, cluster=False,
        mixed_strata=True, n_small=360, k=3, treat_sizes=(1, 1, 1),
        random_state=123,
    )
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.warns(UserWarning, match='Mixed design'):
        fit = sreg(dat.Y, dat.S, dat.D, X=X, HC1=hc1,
                   small_strata=True, k=3)
    assert fit['mixed_design'] is True
    assert fit['res_small'] is not None and fit['res_big'] is not None
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('with_covariates', [True, False])
def test_individual_mixed_data_with_large_option(hc1, with_covariates):
    dat = sreg_rgen(
        600, tau_vec=(.2, .9), n_strata=4, cluster=False,
        mixed_strata=True, n_small=360, k=3, treat_sizes=(1, 1, 1),
        random_state=123,
    )
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.warns(UserWarning, match='At least 25% of strata are small'):
        fit = sreg(dat.Y, dat.S, dat.D, X=X, HC1=hc1, small_strata=False)
    assert fit['small_strata'] is False
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()
    no_strata = sreg(dat.Y, None, dat.D, X=X, HC1=hc1, small_strata=False)
    assert np.isfinite(no_strata['tau_hat']).all()


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('adjustment', [None, ('x_1',), ('x_1', 'x_2')])
def test_cluster_small_data_with_small_option(hc1, adjustment):
    dat = sreg_rgen(120, tau_vec=(.2, .8), cluster=True,
                    small_strata=True, k=3, treat_sizes=(1, 1, 1),
                    random_state=321)
    X = None if adjustment is None else dat[list(adjustment)]
    fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
               HC1=hc1, small_strata=True)
    assert fit['small_strata'] is True
    assert len(fit['tau_hat']) == 2
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('adjustment', [None, ('x_1',), ('x_1', 'x_2')])
def test_cluster_large_data_with_large_option(hc1, adjustment):
    dat = sreg_rgen(200, tau_vec=(.2, .9, 1.5), n_strata=4,
                    cluster=True, random_state=321)
    X = None if adjustment is None else dat[list(adjustment)]
    fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
               HC1=hc1, small_strata=False)
    assert len(fit['tau_hat']) == 3
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('hc1', [True, False])
def test_cluster_small_data_with_large_option_warns(hc1):
    dat = sreg_rgen(120, tau_vec=(.2, .8), cluster=True,
                    small_strata=True, k=3, treat_sizes=(1, 1, 1),
                    random_state=321)
    with pytest.warns(UserWarning) as record:
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng,
                   dat[['x_1', 'x_2']], HC1=hc1, small_strata=False)
    messages = [str(item.message) for item in record]
    assert any('same small number of clusters' in message for message in messages)
    assert any('At least 25% of strata are small' in message for message in messages)
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('with_covariates', [True, False])
def test_cluster_large_data_with_small_option_is_rejected(with_covariates):
    dat = sreg_rgen(200, tau_vec=(.2, .9), n_strata=10,
                    cluster=True, random_state=321)
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.raises(ValueError, match='either all strata are large|too few strata'):
        sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X, small_strata=True)
    with pytest.raises(ValueError, match='Strata indicators are required'):
        sreg(dat.Y, None, dat.D, dat.G_id, dat.Ng, X, small_strata=True)


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('with_covariates', [True, False])
def test_cluster_mixed_data_with_small_option(hc1, with_covariates):
    dat = sreg_rgen(
        240, tau_vec=(.2, .9), n_strata=4, cluster=True,
        mixed_strata=True, n_small=144, k=3, treat_sizes=(1, 1, 1),
        random_state=321,
    )
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.warns(UserWarning, match='Mixed design'):
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
                   HC1=hc1, small_strata=True, k=3)
    assert fit['mixed_design'] is True
    assert fit['res_small'] is not None and fit['res_big'] is not None
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()


@pytest.mark.parametrize('hc1', [True, False])
@pytest.mark.parametrize('with_covariates', [True, False])
def test_cluster_mixed_data_with_large_option(hc1, with_covariates):
    dat = sreg_rgen(
        240, tau_vec=(.2, .9), n_strata=4, cluster=True,
        mixed_strata=True, n_small=144, k=3, treat_sizes=(1, 1, 1),
        random_state=321,
    )
    X = dat[['x_1', 'x_2']] if with_covariates else None
    with pytest.warns(UserWarning, match='At least 25% of strata are small'):
        fit = sreg(dat.Y, dat.S, dat.D, dat.G_id, dat.Ng, X,
                   HC1=hc1, small_strata=False)
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()
    no_strata = sreg(dat.Y, None, dat.D, dat.G_id, dat.Ng, X,
                     HC1=hc1, small_strata=False)
    assert np.isfinite(no_strata['tau_hat']).all()
