# first_test
res_data_gen=sreg_rgen(n=100000, tau_vec=[0, 0.2], cluster=False, is_cov=False)

Y = res_data_gen["Y"]
S = res_data_gen["S"]
D = res_data_gen["D"]

result = sreg(Y = Y, S = S, D = D, G_id=None, Ng=None, X=None, HC1=True)
print(result)

# first_test
res_data_gen=sreg_rgen(n=100000, tau_vec=[5, 0.2], cluster=False, is_cov=True, n_strata = 2)

Y = res_data_gen["Y"]
S = res_data_gen["S"]
D = res_data_gen["D"]
X = res_data_gen[['X1', 'X2']]

result = sreg(Y = Y, S = S, D = D, G_id=None, Ng=None, X=X, HC1=True)
print(result)

# first_test
res_data_gen=sreg_rgen(n=1000, tau_vec=[1, 0.2], cluster=True, Nmax=50, is_cov=True, n_strata = 5)

Y = res_data_gen["Y"]
S = res_data_gen["S"]
D = res_data_gen["D"]
X = res_data_gen[['X1', 'X2', 'Ng']]
G_id = res_data_gen["G_id"]
Ng = res_data_gen["Ng"]

result = sreg(Y = Y, S = S, D = D, G_id=G_id, Ng=Ng, X=X, HC1=True)
print(result)