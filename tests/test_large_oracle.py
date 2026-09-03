import numpy as np
import pandas as pd
import pytest
from sreg import sreg


def individual_data():
    S=np.repeat(np.arange(1,5),36); D=np.tile(np.repeat(np.arange(3),12),4)
    i=np.arange(1,len(S)+1); x1=np.sin(i/7)+S/10; x2=np.cos(i/11)+D/10
    Y=1+.3*x1-.2*x2+.4*(D==1)+.9*(D==2)+.15*S+np.sin(i/5)
    return map(pd.Series,(Y,S,D)),pd.DataFrame({'x1':x1,'x2':x2})


def cluster_data():
    G=np.arange(1,73); S=np.repeat(np.arange(1,5),18); D=np.tile(np.repeat(np.arange(3),6),4)
    ng=2+(G%5); x1=np.sin(G/6); x2=np.cos(G/9)
    y=1+.2*x1-.1*x2+.5*(D==1)+1*(D==2)+.1*S
    gid=np.repeat(G,2); S=np.repeat(S,2); D=np.repeat(D,2); ng=np.repeat(ng,2)
    X=pd.DataFrame({'x1':np.repeat(x1,2),'x2':np.repeat(x2,2)})
    Y=np.repeat(y,2)+np.tile([-.1,.1],72)
    return tuple(map(pd.Series,(Y,S,D,gid,ng))),X


IND={
 (False,False):([-0.0704097795639866,1.17988278892903],[.118520111244193,.124212619570504]),
 (False,True): ([-0.0704097795639866,1.17988278892903],[.12238076187215,.126971899441789]),
 (True,False): ([1.88868802768134,-.424271532937656],[.279647360672016,.245853684988491]),
 (True,True):  ([1.88868802768134,-.424271532937656],[.287301218337414,.252200027859785]),
}
CLU={
 (False,False):([.485767452292169,1.06721724638087],[.149825356758115,.176209756962413]),
 (False,True): ([.485767452292169,1.06721724638087],[.163819335214387,.192829895120849]),
 (True,False): ([8.98068664103738,10.9368408234015],[2.04607878563123,1.35338545255133]),
 (True,True):  ([8.98068664103738,10.9368408234015],[2.19616197095551,1.38821410686241]),
}


@pytest.mark.parametrize('adjusted,hc1',IND)
def test_large_individual_matches_r_2_1(adjusted,hc1):
    values,X=individual_data(); Y,S,D=values
    fit=sreg(Y,S,D,X=X if adjusted else None,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],IND[adjusted,hc1][0],rtol=2e-13)
    np.testing.assert_allclose(fit['se_rob'],IND[adjusted,hc1][1],rtol=2e-13)


@pytest.mark.parametrize('adjusted,hc1',CLU)
def test_large_cluster_matches_r_2_1(adjusted,hc1):
    values,X=cluster_data(); Y,S,D,G,Ng=values
    fit=sreg(Y,S,D,G,Ng,X=X if adjusted else None,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],CLU[adjusted,hc1][0],rtol=2e-12)
    np.testing.assert_allclose(fit['se_rob'],CLU[adjusted,hc1][1],rtol=2e-12)
