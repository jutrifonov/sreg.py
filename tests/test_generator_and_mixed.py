import numpy as np
import pandas as pd
import pytest
from sreg import sreg, sreg_rgen, Sreg


def test_rgen_small_individual_has_exact_allocations():
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,small_strata=True,k=3,
                treat_sizes=(1,1,1),random_state=1)
    assert list(d.columns)==['Y','S','D','x_1','x_2']
    assert (d.groupby(['S','D']).size().unstack(fill_value=0)==1).all().all()


def test_rgen_small_cluster_has_exact_cluster_allocations():
    d=sreg_rgen(60,tau_vec=(.2,.5),cluster=True,small_strata=True,k=3,
                treat_sizes=(1,1,1),random_state=1)
    clusters=d.drop_duplicates('G_id')
    assert len(clusters)==60
    assert (clusters.groupby(['S','D']).size().unstack(fill_value=0)==1).all().all()


def test_mixed_general_k_estimates_and_exposes_components():
    d=sreg_rgen(120,tau_vec=(.5,),cluster=False,mixed_strata=True,n_small=80,
                k=4,treat_sizes=(2,2),n_strata=2,random_state=2)
    with pytest.warns(UserWarning,match='Mixed design'):
        fit=sreg(d.Y,d.S,d.D,small_strata=True,k=4)
    assert fit['mixed_design'] is True
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()
    assert fit['res_small'] is not None and fit['res_big'] is not None


@pytest.mark.parametrize('kwargs,message',[
    ({'n_small':82},'divisible'),
    ({'treat_sizes':(1,1)},'length'),
])
def test_mixed_generator_validation(kwargs,message):
    base=dict(n=120,tau_vec=(.2,.5),cluster=False,mixed_strata=True,n_small=80,
              k=3,treat_sizes=(1,1,1),n_strata=3)
    base.update(kwargs)
    with pytest.raises(ValueError,match=message): sreg_rgen(**base)


def test_custom_large_allocations_and_effects():
    probs=np.array([[.8,.2],[.2,.8]])
    d=sreg_rgen(4000,tau_vec=(.5,),cluster=False,n_strata=2,
                allocation_probs=probs,stratum_effects=(2,-2),
                treatment_effects_by_stratum=np.array([[1.],[3.]]),random_state=4)
    shares=d.groupby('S').D.mean().sort_index().to_numpy()
    np.testing.assert_allclose(shares,[.2,.8],atol=.04)


def test_large_strata_generator_uses_exact_stratum_specific_allocations():
    """Translation of the exact allocation expectation in the R suite."""
    allocation = np.array([
        [.60, .20, .20],
        [.50, .30, .20],
        [.34, .33, .33],
        [.20, .30, .50],
        [.20, .20, .60],
    ])
    generated = sreg_rgen(
        3000, n_strata=5, tau_vec=(.5, .8), cluster=False,
        allocation_probs=allocation, random_state=30,
    )
    counts = pd.crosstab(generated.S, generated.D)
    for stratum in range(1, 6):
        stratum_n = counts.loc[stratum].sum()
        np.testing.assert_array_equal(
            counts.loc[stratum, [1, 2]].to_numpy(),
            np.floor(allocation[stratum - 1, [1, 2]] * stratum_n).astype(int),
        )


def test_large_custom_null_defaults_preserve_generated_data():
    a=sreg_rgen(200,tau_vec=(.5,),cluster=False,n_strata=4,random_state=91)
    b=sreg_rgen(200,tau_vec=(.5,),cluster=False,n_strata=4,
                allocation_probs=None,stratum_effects=None,
                treatment_effects_by_stratum=None,random_state=91)
    assert a.equals(b)


def test_default_large_allocation_gives_remainders_to_control_like_r():
    generated=sreg_rgen(103,n_strata=4,tau_vec=(.2,.8),cluster=False,
                        random_state=31)
    counts=pd.crosstab(generated.S,generated.D)
    for _,row in counts.iterrows():
        expected_active=int(np.floor(row.sum()/3))
        assert row[1]==expected_active and row[2]==expected_active
        assert row[0]==row.sum()-2*expected_active


def test_nonmixed_small_omitted_treatment_sizes_uses_r_literal_default():
    with pytest.raises(ValueError,match='treat_sizes'):
        sreg_rgen(40,tau_vec=(.5,),cluster=False,small_strata=True,k=2,
                  random_state=32)


def test_generator_uses_gamma_vector_and_r_cluster_size_support():
    base=sreg_rgen(400,tau_vec=(.5,),cluster=False,is_cov=True,
                   gamma_vec=(0,0,0),random_state=33)
    shifted=sreg_rgen(400,tau_vec=(.5,),cluster=False,is_cov=True,
                      gamma_vec=(.4,.2,1),random_state=33)
    assert not np.allclose(base.Y,shifted.Y)
    clusters=sreg_rgen(200,tau_vec=(.5,),cluster=True,Nmax=50,
                       random_state=34).drop_duplicates('G_id')
    assert set(clusters.Ng.unique()).issubset({10,20,30,40,50})
    without=sreg_rgen(40,tau_vec=(.5,),cluster=True,is_cov=False,
                      random_state=35)
    assert 'x_1' not in without and 'x_2' not in without


def test_custom_large_arguments_are_validated():
    with pytest.raises(ValueError,match='allocation_probs'):
        sreg_rgen(100,tau_vec=(.5,),cluster=False,n_strata=2,
                  allocation_probs=np.array([[.5,.5],[.2,.2]]))
    with pytest.raises(ValueError,match='stratum_effects'):
        sreg_rgen(100,tau_vec=(.5,),cluster=False,n_strata=2,stratum_effects=(1,))
    with pytest.raises(ValueError,match='treatment_effects_by_stratum'):
        sreg_rgen(100,tau_vec=(.5,),cluster=False,n_strata=2,
                  treatment_effects_by_stratum=np.ones((2,2)))
    with pytest.raises(ValueError,match='supported only'):
        sreg_rgen(100,tau_vec=(.5,),cluster=True,n_strata=2,stratum_effects=(1,2))


def test_mixed_omitted_treatment_sizes_derives_even_allocation():
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,mixed_strata=True,n_small=72,
                k=4,n_strata=3,random_state=12)
    small=d[d.S<=18]
    allocation=small.groupby(['S','D']).size().unstack(fill_value=0)
    assert (allocation.sum(axis=1)==4).all()
    assert set(allocation.iloc[0])=={1,2}


def test_plot_returns_matplotlib_axes():
    d=sreg_rgen(120,tau_vec=(.5,),cluster=False,n_strata=4,random_state=5)
    fit=sreg(d.Y,d.S,d.D)
    assert isinstance(fit,Sreg)
    assert fit.plot().__class__.__name__=='Axes'


@pytest.mark.parametrize('supply_ng',[True,False])
def test_mixed_cluster_population_weights_and_share_variance(supply_ng):
    d=sreg_rgen(120,tau_vec=(.4,.9),n_strata=4,cluster=True,mixed_strata=True,
                n_small=72,k=3,treat_sizes=(1,1,1),random_state=3)
    ng=d.Ng.copy() if supply_ng else None
    if supply_ng:
        small=d.S.le(24); ng.loc[small] *= 2
    with pytest.warns(UserWarning,match='Mixed design'):
        fit=sreg(d.Y,d.S,d.D,d.G_id,ng,small_strata=True,k=3,HC1=False)
    clusters=fit['data'][['G_id','stratum_type','Ng']].drop_duplicates('G_id')
    ns=clusters.loc[clusters.stratum_type.eq('small'),'Ng'].sum(); total=clusters.Ng.sum()
    ps=ns/total; pb=1-ps
    expected=ps*fit['res_small']['tau_hat']+pb*fit['res_big']['tau_hat']
    np.testing.assert_allclose(fit['tau_hat'],expected)
    g=len(clusters); nbar=total/g; is_big=clusters.stratum_type.eq('big').astype(float)
    vp=np.mean(clusters.Ng**2*(is_big-pb)**2)/nbar**2
    expected_var=ps**2*fit['res_small']['se_rob']**2+pb**2*fit['res_big']['se_rob']**2
    expected_var += vp/g*(fit['res_small']['tau_hat']-fit['res_big']['tau_hat'])**2
    np.testing.assert_allclose(fit['se_rob']**2,expected_var)


@pytest.mark.parametrize('cluster',[False,True])
def test_mixed_adjustment_is_applied_to_both_components(cluster):
    d=sreg_rgen(180,tau_vec=(.5,),cluster=cluster,mixed_strata=True,n_small=120,
                k=3,treat_sizes=(1,2),n_strata=3,random_state=44)
    args=(d.Y,d.S,d.D,d.G_id,d.Ng) if cluster else (d.Y,d.S,d.D)
    with pytest.warns(UserWarning,match='Mixed design'):
        fit=sreg(*args,X=d[['x_1']],small_strata=True,k=3,HC1=False)
    assert fit['res_small']['lin_adj'] is not None
    assert fit['res_big']['lin_adj'] is not None
    assert fit['res_small']['lin_adj'].shape[1]==1
    assert fit['res_big']['lin_adj'].shape[1]==1


def test_is_cov_controls_large_cluster_columns():
    a=sreg_rgen(20,n_strata=2,tau_vec=(.5,),cluster=True,is_cov=False,random_state=99)
    b=sreg_rgen(20,n_strata=2,tau_vec=(.5,),cluster=True,is_cov=True,random_state=99)
    assert {'Y','S','D','G_id','Ng'} <= set(a.columns)
    assert not {'x_1','x_2'} & set(a.columns)
    assert {'x_1','x_2'} <= set(b.columns)


@pytest.mark.parametrize('cluster,n,n_small,n_strata',[(False,120,80,4),(True,80,48,3)])
def test_general_four_tuple_mixed_design_structure(cluster,n,n_small,n_strata):
    d=sreg_rgen(n,tau_vec=(.5,),cluster=cluster,mixed_strata=True,n_small=n_small,
                k=4,treat_sizes=(2,2),n_strata=n_strata,random_state=101)
    args=(d.Y,d.S,d.D,d.G_id,d.Ng) if cluster else (d.Y,d.S,d.D)
    with pytest.warns(UserWarning,match='k = 4'):
        fit=sreg(*args,small_strata=True,k=4)
    small=fit['res_small']['data']
    sizes=(small[['S','G_id']].drop_duplicates().groupby('S').size() if cluster else small.groupby('S').size())
    assert fit['mixed_design'] and (sizes==4).all()


def test_general_k_is_required_when_automatic_detection_cannot_find_it():
    d=sreg_rgen(120,tau_vec=(.5,),cluster=False,mixed_strata=True,n_small=80,
                k=4,treat_sizes=(2,2),n_strata=4,random_state=102)
    with pytest.raises(ValueError,match='too few strata'):
        sreg(d.Y,d.S,d.D,small_strata=True)
    with pytest.raises(ValueError,match='too few strata'):
        sreg(d.Y,d.S,d.D,small_strata=True,k=5)


@pytest.mark.parametrize('cluster',[False,True])
def test_mixed_unidentified_large_adjustment_has_actionable_error(cluster):
    d=sreg_rgen(90,tau_vec=(.5,),cluster=cluster,mixed_strata=True,n_small=60,
                k=3,treat_sizes=(1,2),n_strata=2,random_state=303)
    rng=np.random.default_rng(1); X=pd.DataFrame(rng.normal(size=(len(d),12)))
    args=(d.Y,d.S,d.D,d.G_id,d.Ng) if cluster else (d.Y,d.S,d.D)
    with pytest.raises(ValueError,match='large-strata component.*cannot support'):
        with pytest.warns(UserWarning,match='Mixed design'):
            sreg(*args,X=X,small_strata=True,k=3)
    with pytest.warns(UserWarning,match='Mixed design'):
        fit=sreg(*args,X=None,small_strata=True,k=3)
    assert np.isfinite(fit['tau_hat']).all()
