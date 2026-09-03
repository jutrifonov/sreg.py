import numpy as np
import pandas as pd
import pytest
from sreg import sreg, sreg_rgen


@pytest.fixture
def sample():
    return sreg_rgen(120,tau_vec=(.2,.5),cluster=False,n_strata=4,random_state=7)


@pytest.mark.parametrize('hc',[5,'True',None])
def test_hc1_requires_boolean(sample,hc):
    with pytest.raises(ValueError,match='value of HC'):
        sreg(sample.Y,sample.S,sample.D,HC1=hc)


@pytest.mark.parametrize('field',[0,1,2])
def test_integer_design_variables(sample,field):
    values=[sample.S.astype(float).copy(),sample.D.astype(float).copy(),pd.Series(np.arange(len(sample)),dtype=float)]
    values[field].iloc[0]=1.5
    with pytest.raises(ValueError,match='integer values'):
        sreg(sample.Y,values[0],values[1],values[2] if field==2 else None)


@pytest.mark.parametrize('bad',[[1.,2.],np.array(['a','b'])])
def test_rejects_non_numeric_or_list_inputs(sample,bad):
    with pytest.raises(ValueError,match='different type'):
        sreg(bad,sample.S.iloc[:2],sample.D.iloc[:2])


def test_missing_outcome_and_treatment(sample):
    with pytest.raises(ValueError,match='Observed outcomes'):
        sreg(None,sample.S,sample.D)
    with pytest.raises(ValueError,match='Treatments'):
        sreg(sample.Y,sample.S,None)


def test_nan_rows_warn_and_are_omitted(sample):
    y=sample.Y.copy(); y.iloc[:3]=np.nan
    with pytest.warns(UserWarning,match='ignoring these values'):
        fit=sreg(y,sample.S,sample.D)
    assert len(fit['data'])==len(sample)-3


def test_skipped_strata_and_treatments_are_rejected(sample):
    S=sample.S.replace(3,2)
    with pytest.raises(ValueError,match='skipped values'):
        sreg(sample.Y,S,sample.D)
    D=sample.D.replace(1,0)
    with pytest.raises(ValueError,match='skipped values'):
        sreg(sample.Y,sample.S,D)


def test_indices_start_at_expected_values(sample):
    with pytest.raises(ValueError,match='strata should be indexed'):
        sreg(sample.Y,sample.S-1,sample.D)
    with pytest.raises(ValueError,match='treatments should be indexed'):
        sreg(sample.Y,sample.S,sample.D+1)


def test_cluster_level_variables_must_be_constant():
    d=sreg_rgen(30,tau_vec=(.5,),cluster=True,n_strata=3,random_state=8)
    S=d.S.copy(); first=d.G_id.eq(d.G_id.iloc[0]); S.loc[first.idxmax()]=2 if S.iloc[0]!=2 else 1
    with pytest.raises(ValueError,match='consistent within each cluster'):
        sreg(d.Y,S,d.D,d.G_id,d.Ng)


def test_missing_cluster_sizes_warns_and_infers():
    d=sreg_rgen(30,tau_vec=(.5,),cluster=True,n_strata=3,random_state=8)
    with pytest.warns(UserWarning,match='Cluster sizes have not been provided'):
        fit=sreg(d.Y,d.S,d.D,d.G_id,None)
    assert np.isfinite(fit['tau_hat']).all()


def test_small_strata_requires_strata(sample):
    with pytest.raises(ValueError,match='Strata indicators'):
        sreg(sample.Y,None,sample.D,small_strata=True)
