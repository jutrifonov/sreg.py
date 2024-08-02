import pandas as pd
import numpy as np
# The core

def sreg(Y, S=None, D=None, G_id=None, Ng=None, X=None, HC1=True):
    check_data_types(Y, S, D, G_id, Ng, X)
    check_integers(S, D, G_id, Ng)
    boolean_check(HC1)

    if Y is None:
        raise ValueError("Error: Observed outcomes have not been provided (Y = None). Please provide the vector of observed outcomes.")
    if D is None:
        raise ValueError("Error: Treatments have not been provided (D = None). Please provide the vector of treatments.")
    if D is not None:
        check_range(D)
    if S is not None:
        check_range(S)

    if X is not None and 'Ng' in X.columns:
        X = X.rename(columns={'Ng': 'Ng_1'})
    if X is not None:
        if isinstance(X, pd.DataFrame):
            X_dict = {col: X[col].values for col in X.columns}
        elif isinstance(X, np.ndarray):
            X_dict = {f'X{i}': X[:, i] for i in range(X.shape[1])}
        else:
            raise ValueError("X should be a pandas DataFrame or a NumPy array.")
    else:
        X_dict = {}
    #if Ng is not None:
    check_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng, **X_dict})
    if G_id is None: 
        check_df=check_df.drop(columns=['G_id'])
    #else:
    #    check_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, **X_dict})

    if S is None:
        check_df=check_df.drop(columns=['S'])
    if Ng is None:
        check_df=check_df.drop(columns=['Ng'])
    #check_df = pd.DataFrame({'Y': Y, 'S': S, 'D': D, 'G_id': G_id, 'Ng': Ng, **{f'X{i}': X[:, i] for i in range(X.shape[1])} if X is not None else {}})

    if check_df.isnull().any().any():
        print("Warning: The data contains one or more NA (or NaN) values. Proceeding while ignoring these values.")
    clean_df = check_df.dropna()

    x_ind = max([i for i, col in enumerate(clean_df.columns) if col in ['D', 'G_id', 'Ng']], default=-1)

    Y = clean_df['Y']
    if S is not None:
        S = clean_df['S']
    D = clean_df['D']
    if G_id is not None:
        G_id = clean_df['G_id']
    if Ng is not None:
        Ng = clean_df['Ng']
    
    if (x_ind + 1) >= len(clean_df.columns):
        X = None
    else:
        X = clean_df.iloc[:, (x_ind + 1):]

    if X is not None and 'Ng_1' in X.columns:
        X = X.rename(columns={'Ng_1': 'Ng'})

    if S is not None:
        if S.min() != 1:
            raise ValueError(f"Error: The strata should be indexed by {{1, 2, 3, ...}}. The minimum value in the provided data is {S.min()}.")
    if D is not None:
        if D.min() != 0:
            raise ValueError(f"Error: The treatments should be indexed by {{0, 1, 2, ...}}, where D = 0 denotes the control. The minimum value in the provided data is {D.min()}.")

    if G_id is None:
        result = res_sreg(Y, S, D, X, HC1)
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
    else:
        check_cluster_lvl(G_id, S, D, Ng)
        result = res_creg(Y, S, D, G_id, Ng, X, HC1)
        if Ng is None:
            print("Warning: Cluster sizes have not been provided (Ng = None). Ng is assumed to be equal to the number of available observations in every cluster g.")
        if X is not None:
            if any(np.isnan(x).any() for x in result['ols_iter']):
                raise ValueError("Error: There are too many covariates relative to the number of observations. Please reduce the number of covariates (k = ncol(X)) or consider estimating the model without covariate adjustments.")
        if result['lin_adj'] is not None:
            if not check_cluster(pd.DataFrame({'G_id': result['data']['G_id'], **result['lin_adj']})):
                print("Warning: sreg cannot use individual-level covariates for covariate adjustment in cluster-randomized experiments. Any individual-level covariates have been aggregated to their cluster-level averages.")

    return result

# S[2]=0.23
# G_id[5]=8.98
# resultim=sreg(Y=Y, S=None, D=D, G_id=G_id, Ng=Ng, X=X, HC1=True)
# print(resultim)
# Ng = None
# S=None
# D[2]=1.2

# aejapp=sreg(Y=Y, S=S, D=D, X=X)
# print(aejapp)
# X=None
# print(aejapp)
# Ng=None
# G_id=None
# G_id
# X=None