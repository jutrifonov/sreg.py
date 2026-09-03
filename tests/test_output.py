import matplotlib
matplotlib.use('Agg')
from sreg import sreg,sreg_rgen


def test_print_large_strata_metadata(capsys):
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,n_strata=4,random_state=21)
    fit=sreg(d.Y,d.S,d.D,X=d[['x_1','x_2']])
    print(fit); out=capsys.readouterr().out
    for text in ['Saturated Model Estimation Results under CAR with linear adjustments',
                 'Observations: 120','Number of treatments: 2','Number of strata: 4',
                 'Setup: large strata','Standard errors: adjusted (HC1)',
                 'Treatment assignment: individual level',
                 'Covariates used in linear adjustments: x_1, x_2','Signif. codes:']:
        assert text in out


def test_print_small_cluster_metadata(capsys):
    d=sreg_rgen(32,tau_vec=(.5,),cluster=True,small_strata=True,k=2,
                treat_sizes=(1,1),random_state=22)
    fit=sreg(d.Y,d.S,d.D,d.G_id,d.Ng,small_strata=True,HC1=False)
    print(fit); out=capsys.readouterr().out
    for text in ['Clusters: 32','Setup: small strata','Strata size (k): 2',
                 'Standard errors: unadjusted','Treatment assignment: cluster level']:
        assert text in out


def test_print_large_cluster_and_unadjusted_metadata(capsys):
    d=sreg_rgen(200,tau_vec=(.2,.5,.9),cluster=True,n_strata=6,random_state=24)
    adjusted=sreg(d.Y,d.S,d.D,d.G_id,d.Ng,d[['x_1','x_2']],HC1=False)
    print(adjusted); out=capsys.readouterr().out
    for text in ['Saturated Model Estimation Results under CAR with linear adjustments',
                 'Clusters: 200','Number of treatments: 3','Number of strata: 6',
                 'Setup: large strata','Standard errors: unadjusted',
                 'Treatment assignment: cluster level',
                 'Covariates used in linear adjustments: x_1, x_2','Signif. codes:']:
        assert text in out
    unadjusted=sreg(d.Y,d.S,d.D,d.G_id,d.Ng,None,HC1=True)
    print(unadjusted); out=capsys.readouterr().out
    for text in ['Saturated Model Estimation Results under CAR',
                 'Clusters: 200','Standard errors: adjusted (HC1)',
                 'Covariates used in linear adjustments: ']:
        assert text in out


def test_print_mixed_cluster_metadata(capsys):
    d=sreg_rgen(240,tau_vec=(.2,.5),cluster=True,n_strata=4,
                mixed_strata=True,n_small=144,k=3,
                treat_sizes=(1,1,1),random_state=25)
    fit=sreg(d.Y,d.S,d.D,d.G_id,d.Ng,d[['x_1']],
             small_strata=True,k=3)
    print(fit); out=capsys.readouterr().out
    for text in ['Clusters: 240','Number of treatments: 2',
                 'Setup: mixed design (includes both small and large strata)',
                 'Strata size (k, small strata only): 3',
                 'Standard errors: adjusted (HC1)',
                 'Treatment assignment: cluster level',
                 'Covariates used in linear adjustments: x_1','Signif. codes:']:
        assert text in out


def test_plot_customization_returns_axes():
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,n_strata=4,random_state=23)
    fit=sreg(d.Y,d.S,d.D)
    ax=fit.plot(treatment_labels=['A','B'],title='Effects',bar_fill='purple',
                grid=False,zero_line=False,y_axis_title='Programs',x_axis_title='ATE')
    assert ax.get_title()=='Effects' and ax.get_xlabel()=='ATE' and ax.get_ylabel()=='Programs'
    assert [x.get_text() for x in ax.get_yticklabels()]==['A','B']


def test_plot_supports_all_r_style_options_and_annotations():
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,n_strata=4,random_state=26)
    fit=sreg(d.Y,d.S,d.D)
    ax=fit.plot(bar_fill=['navy','gold'],point_shape=23,point_size=4,
                point_fill='white',point_stroke=2,point_color='red',
                label_color='green',label_size=8,bg_color='#eeeeee')
    assert len(ax.texts)==2
    assert all('(' in text.get_text() for text in ax.texts)
    assert all(text.get_color()=='green' for text in ax.texts)


def test_plot_invalid_bar_fill_warns_and_falls_back():
    import pytest
    d=sreg_rgen(120,tau_vec=(.2,.5),cluster=False,n_strata=4,random_state=27)
    fit=sreg(d.Y,d.S,d.D)
    with pytest.warns(UserWarning,match='bar_fill must be'):
        ax=fit.plot(bar_fill=['red','green','blue'])
    assert ax is not None
