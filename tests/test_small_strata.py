import numpy as np
import pandas as pd
import pytest
from sreg import sreg


def test_multiarm_small_strata_matches_r_2_1_oracle():
    # Oracle: sreg 2.1.0, HC1=FALSE. Values are generated in tests/oracle.R.
    y=np.array([1,2,4, 2,4,5, 3,6,8, 4,7,10.])
    S=np.repeat(np.arange(1,5),3); D=np.tile(np.arange(3),4)
    fit=sreg(y,S,D,small_strata=True,HC1=False)
    np.testing.assert_allclose(fit['tau_hat'],[2.25,4.25],rtol=1e-12)
    np.testing.assert_allclose(fit['se_rob'],[.590726953281576,.657488909919146],rtol=1e-12)


def test_small_strata_requires_even_number_for_paired_variance():
    with pytest.raises(ValueError,match='even number'):
        sreg(np.arange(9.),np.repeat(np.arange(1,4),3),np.tile(np.arange(3),3),small_strata=True)


def test_explicit_k_is_validated():
    y=np.arange(12.); S=np.repeat(np.arange(1,5),3); D=np.tile(np.arange(3),4)
    with pytest.raises(ValueError,match='does not match'):
        sreg(y,S,D,small_strata=True,k=4)
    with pytest.raises(ValueError,match='positive integer'):
        sreg(y,S,D,small_strata=True,k=0)


def test_small_cluster_ng_null_equals_observed_sizes():
    gid=np.repeat(np.arange(1,13),2); S=np.repeat(np.repeat(np.arange(1,5),3),2)
    D=np.repeat(np.tile(np.arange(3),4),2); y=np.arange(24.)
    a=sreg(y,S,D,gid,np.repeat(2,24),small_strata=True,HC1=False)
    b=sreg(y,S,D,gid,None,small_strata=True,HC1=False)
    np.testing.assert_allclose(a['tau_hat'],b['tau_hat'])
    np.testing.assert_allclose(a['se_rob'],b['se_rob'])


def test_multiarm_small_cluster_matches_r_2_1_oracle():
    gid=np.arange(1,13); S=np.repeat(np.arange(1,5),3); D=np.tile(np.arange(3),4)
    Ng=np.tile([2,3,4],4); y=np.array([1,2,4,2,4,5,3,6,8,4,7,10.])
    fit=sreg(y,S,D,gid,Ng,small_strata=True,HC1=False)
    np.testing.assert_allclose(fit['tau_hat'],[3.08333333333333,7.33333333333333],rtol=1e-12)
    np.testing.assert_allclose(fit['se_rob'],[.602368012283373,.95257934441568],rtol=1e-12)


def test_small_covariate_adjustment_is_finite():
    y=np.array([1,2,4,2,4,5,3,6,8,4,7,10.]); S=np.repeat(np.arange(1,5),3); D=np.tile(np.arange(3),4)
    X=pd.DataFrame({'x':np.array([0,1,3,1,2,4,2,4,7,3,6,9.])})
    fit=sreg(y,S,D,X=X,small_strata=True,HC1=False)
    assert np.isfinite(fit['tau_hat']).all() and np.isfinite(fit['se_rob']).all()
    assert fit['beta_hat'].shape==(2,1)
