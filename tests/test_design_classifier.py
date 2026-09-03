import numpy as np
import pandas as pd
import pytest
from sreg.small_strata import classify_strata


def frame_for_sizes(sizes):
    S=np.concatenate([np.repeat(i+1,n) for i,n in enumerate(sizes)])
    return pd.DataFrame({'Y':np.arange(len(S),dtype=float),'S':S,'D':np.arange(len(S))%2})


def test_mixed_warning_reports_detected_individual_k():
    data=frame_for_sizes([4,4,4,4,20,21,22,23])
    with pytest.warns(UserWarning,match='k = 4'):
        out=classify_strata(data,small_strata=True,k=4)
    assert set(out.stratum_type)=={'small','big'}


def test_mixed_warning_detects_k_from_cluster_counts():
    clusters=frame_for_sizes([3,3,3,3,10,11,12,13]).rename(columns={'Y':'G_id'})
    clusters['Y']=clusters.G_id.astype(float)
    with pytest.warns(UserWarning,match='k = 3'):
        out=classify_strata(clusters,cluster=True,small_strata=True)
    assert out.loc[out.S.le(4),'stratum_type'].eq('small').all()


def test_less_than_quarter_at_one_size_is_not_mixed_candidate():
    data=frame_for_sizes([3,10,11,12,13])
    with pytest.raises(ValueError,match='too few strata'):
        classify_strata(data,small_strata=True)


def test_large_option_warns_at_25_percent_small_strata():
    data=frame_for_sizes([3,3,10,11,12,13,14,15])
    with pytest.warns(UserWarning,match='25%'):
        classify_strata(data,small_strata=False)
