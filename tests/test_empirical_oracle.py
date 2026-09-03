import numpy as np
from sreg import AEJapp,sreg


def test_aej_example_matches_r_2_1_oracle():
    data=AEJapp(); D=data.treatment.replace(3,0)
    plain=sreg(data.gradesq34,data.class_level,D)
    adjusted=sreg(data.gradesq34,data.class_level,D,X=data[['pills_taken','age_months']])
    np.testing.assert_allclose(plain['tau_hat'],[-.05112971,.40903373],atol=5e-9)
    np.testing.assert_allclose(plain['se_rob'],[.2064541,.2065146],atol=5e-8)
    np.testing.assert_allclose(adjusted['tau_hat'],[-.02861589,.34608688],atol=5e-9)
    np.testing.assert_allclose(adjusted['se_rob'],[.1816173,.1857249],atol=5e-8)


def test_aejapp_is_declared_public_api():
    import sreg as package
    assert 'AEJapp' in package.__all__
