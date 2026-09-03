import numpy as np
import pandas as pd
import pytest
from sreg import sreg


def data(cluster=False):
    S=np.repeat(np.arange(1,9),3); D=np.tile(np.arange(3),8); i=np.arange(1,25); x=np.sin(i/5)
    Y=1+.2*x+.5*(D==1)+.9*(D==2)+.1*S+np.cos(i/4)
    base=tuple(map(pd.Series,(Y,S,D)))
    if cluster:
        G=np.arange(1,25); Ng=2+(G%4); base+=tuple(map(pd.Series,(G,Ng)))
    return base,pd.DataFrame({'x':x})


IND={
 (False,False):([.489892132015968,.881414901771266],[.144536000057893,.155812548234897]),
 (False,True): ([.489892132015968,.881414901771266],[.144536000057893,.155812548234897]),
 (True,False): ([.459775180791031,.811728129098931],[.11269392470557,.121328501148711]),
 (True,True):  ([.459775180791031,.811728129098931],[.136679078289904,.143799699497835]),
}
CLU={
 (False,False):([.484938997257771,.841484649986121],[.378787718498259,.30995558524523]),
 (False,True): ([.484938997257771,.841484649986121],[.409265707152126,.325218145197007]),
 (True,False): ([.531267178917336,.902872307825754],[.444733800460319,.335506768240645]),
 (True,True):  ([.531267178917336,.902872307825754],[.530514908643059,.379432102378508]),
}


@pytest.mark.parametrize('adjusted,hc1',IND)
def test_small_individual_matches_r_2_1(adjusted,hc1):
    values,X=data(); fit=sreg(*values,X=X if adjusted else None,small_strata=True,k=3,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],IND[adjusted,hc1][0],rtol=2e-12)
    np.testing.assert_allclose(fit['se_rob'],IND[adjusted,hc1][1],rtol=2e-12)


@pytest.mark.parametrize('adjusted,hc1',CLU)
def test_small_cluster_matches_r_2_1(adjusted,hc1):
    values,X=data(True); fit=sreg(*values,X=X if adjusted else None,small_strata=True,k=3,HC1=hc1)
    np.testing.assert_allclose(fit['tau_hat'],CLU[adjusted,hc1][0],rtol=2e-12)
    np.testing.assert_allclose(fit['se_rob'],CLU[adjusted,hc1][1],rtol=2e-12)
