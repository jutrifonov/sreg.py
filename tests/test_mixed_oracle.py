import warnings
import numpy as np
import pandas as pd
import pytest
from sreg import sreg


def individual():
    S=np.r_[np.repeat(np.arange(1,9),3),np.repeat(np.arange(9,11),18)]
    D=np.r_[np.tile(np.arange(3),8),np.tile(np.repeat(np.arange(3),6),2)]
    i=np.arange(1,len(S)+1); x=np.sin(i/8)+S/20
    Y=1+.3*x+.6*(D==1)+1.1*(D==2)+.05*S+np.cos(i/6)
    return tuple(map(pd.Series,(Y,S,D))),pd.DataFrame({'x':x})


def cluster():
    S=np.r_[np.repeat(np.arange(1,9),3),np.repeat(np.arange(9,11),9)]
    D=np.r_[np.tile(np.arange(3),8),np.tile(np.repeat(np.arange(3),3),2)]
    G=np.arange(1,len(S)+1); Ng=2+(G%5); x=np.sin(G/7)
    Y=1+.25*x+.5*(D==1)+.95*(D==2)+.08*S+np.cos(G/9)
    return tuple(map(pd.Series,(Y,S,D,G,Ng))),pd.DataFrame({'x':x})


IND={
 (False,False):([.5737445189471,1.05623315611625],[.0874058554130848,.0973301686890425]),
 (False,True): ([.5737445189471,1.05623315611625],[.0897184558385744,.0989502665887475]),
 (True,False): ([.634523861202921,.94545191026228],[.0765954003232017,.0917911481039153]),
 (True,True):  ([.634523861202921,.94545191026228],[.0789324690961289,.0940247512401473]),
}
CLU={
 (False,False):([.6042489018932,.992742559219785],[.26554831140307,.289978138637724]),
 (False,True): ([.6042489018932,.992742559219785],[.289121706384732,.315964235502771]),
 (True,False): ([.889379471370624,.0228832875447843],[.335895307464494,.420425085719461]),
 (True,True):  ([.889379471370624,.0228832875447843],[.390182437234945,.460089318134126]),
}


@pytest.mark.parametrize('adjusted,hc1',IND)
def test_mixed_individual_matches_r_2_1(adjusted,hc1):
    values,X=individual()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); fit=sreg(*values,X=X if adjusted else None,small_strata=True,k=3,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],IND[adjusted,hc1][0],rtol=2e-12)
    np.testing.assert_allclose(fit['se_rob'],IND[adjusted,hc1][1],rtol=2e-12)


@pytest.mark.parametrize('adjusted,hc1',CLU)
def test_mixed_cluster_matches_r_2_1(adjusted,hc1):
    values,X=cluster()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); fit=sreg(*values,X=X if adjusted else None,small_strata=True,k=3,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],CLU[adjusted,hc1][0],rtol=2e-12)
    np.testing.assert_allclose(fit['se_rob'],CLU[adjusted,hc1][1],rtol=2e-12)
