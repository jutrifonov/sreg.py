import numpy as np
import pandas as pd
from sreg.var_hat import as_var_creg


def fixture():
    S=np.repeat([1,2],9); D=np.tile(np.repeat([0,1,2],3),2)
    A=np.where(D==1,1,np.where(D==0,0,-999999))
    Ng=np.array([2,4,7,3,6,8,5,9,11,3,5,8,4,7,10,6,9,13])
    total=1.5+.4*Ng+.8*(D==1)+1.3*(D==2)+.5*(S==2)
    pi=np.repeat(1/3,len(D)); mu0=.2+.15*Ng+.1*(S==2); mud=.6+.25*Ng+.2*(S==2)
    data=pd.DataFrame({'S':S,'D':D,'A':A,'I':np.isin(D,[0,1]).astype(int),'Ng':Ng})
    return {'tau_hat':np.array([.35]),'Y_bar_g':total/Ng,'Ng':Ng,
            'mu_hat':[np.c_[mu0,mud]],'pi_hat':[pi],'pi_hat_0':pi,'data_list':[data]}


def test_adjusted_cluster_variance_includes_other_arms_r_oracle():
    np.testing.assert_allclose(as_var_creg(True,fixture(),False),[.05866999],atol=1e-8)


def test_unadjusted_cluster_variance_includes_other_arms_r_oracle():
    np.testing.assert_allclose(as_var_creg(None,fixture(),False),[.08752711],atol=1e-8)
