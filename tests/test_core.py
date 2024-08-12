import pytest
import numpy as np
import pandas as pd
from sreg import sreg, sreg_rgen

def test_sreg():
    Y = np.random.normal(size=100)
    S = np.random.choice([1, 2, 3, 4, 5], size=100, replace=True)
    D = np.random.choice([0, 1], size=100, replace=True)
    result = sreg(Y, S, D)
    
    assert len(result['tau_hat']) == 1
    assert result['tau_hat'][0] > 0



def test_simulations_without_clusters():
    #np.random.seed(123)
    data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg_py/src/sreg/data_test_1.csv")
    Y = data['Y']
    S = data['S']
    D = data['D']
    X = pd.DataFrame({'x_1': data['x_1'], 'x_2': data['x_2']})

    result = sreg(Y, S, D, G_id=None, Ng=None, X=X)
    assert np.allclose(np.round(result['tau_hat'], 7), [0.1580814, 0.4846882])
    assert np.allclose(np.round(result['se_rob'], 8), [0.07524021, 0.07616346])

    result = sreg(Y, S, D, G_id=None, Ng=None, X=None)
    assert np.allclose(np.round(result['tau_hat'], 7), [0.1627114, 0.4948722])
    assert np.allclose(np.round(result['se_rob'], 7), [0.1105611, 0.1124072])

    result = sreg(Y, S=None, D=D, G_id=None, Ng=None, X=X)
    assert np.allclose(np.round(result['tau_hat'], 7), [0.1578917, 0.4963735])
    assert np.allclose(np.round(result['se_rob'], 8), [0.08255663, 0.08320655])

    result = sreg(Y, S=None, D=D, G_id=None, Ng=None, X=None)
    assert np.allclose(np.round(result['tau_hat'], 7), [0.1685108, 0.5022035])
    assert np.allclose(np.round(result['se_rob'], 7), [0.1145915, 0.1161482])

    with pytest.raises(ValueError, match="The value of HC must be either True or False."):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X, HC1=5)

    with pytest.raises(ValueError, match="The value of HC must be either True or False."):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X, HC1="True")

    with pytest.raises(ValueError, match="variable has a different type than matrix, numeric vector, or data frame."):
        sreg(list(Y), S=S, D=D, G_id=None, Ng=None, X=X)

    S[2] = 2.5
    with pytest.raises(ValueError, match="must contain only integer values."):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=None)

    S = data['S']
    D[5] = 0.5
    with pytest.raises(ValueError, match="must contain only integer values."):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    S[3] = 5.5
    with pytest.raises(ValueError, match="must contain only integer values."):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    S = data['S']
    D = data['D']

    X.iloc[10:12, :] = np.nan
    Y[:10] = np.nan
    msg = "ignoring these values"
    with pytest.warns(UserWarning, match=msg):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X, HC1=True)

    X = pd.DataFrame({'x_1': data['X1'], 'x_2': data['X2']})
    Y = data['Y']
    S[12] = np.nan
    with pytest.warns(UserWarning, match=msg):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X, HC1=True)

    S = data['S']
    S[19] = np.nan
    with pytest.warns(UserWarning, match=msg):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X, HC1=True)

    S = data['S']
    S[1] = 0
    with pytest.raises(ValueError, match="The strata should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    S[10] = -1
    with pytest.raises(ValueError, match="The strata should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    with pytest.raises(ValueError, match="The strata should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=None)

    assert sreg(Y, S=None, D=D, G_id=None, Ng=None, X=X)

    S = data['S']
    D[:3] = -1
    with pytest.raises(ValueError, match="The treatments should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    D[4] = -2
    with pytest.raises(ValueError, match="The treatments should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)

    D = data['D']
    D[0] = -1
    with pytest.raises(ValueError, match="The treatments should be indexed by"):
        sreg(Y, S=None, D=D, G_id=None, Ng=None, X=X)

    with pytest.raises(ValueError, match="The treatments should be indexed by"):
        sreg(Y, S=S, D=D, G_id=None, Ng=None, X=None)

    with pytest.raises(ValueError, match="The treatments should be indexed by"):
        sreg(Y, S=None, D=D, G_id=None, Ng=None, X=None)

    D = data['D']
    assert sreg(Y, S=S, D=D, G_id=None, Ng=None, X=X)
