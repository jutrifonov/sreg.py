import pandas as pd
#-------------------------------------------------------------------------
# %#     Function to check if covariates are the same within each cluster
#-------------------------------------------------------------------------
def check_cluster(df):
    cov_names = df.columns[1:]
    cov_same = df.groupby('G_id').agg({col: lambda x: x.nunique() == 1 for col in cov_names}).reset_index()
    all_same = cov_same[cov_names].all(axis=1).all()
    return bool(all_same)

 #df = pd.concat([G_id, X], axis=1)
 #check_cluster(df)

def check_cluster_lvl(G_id, S, D, Ng):
    dta_check_lvl = pd.DataFrame({'G_id': G_id, 'S': S, 'D': D, 'Ng': Ng})
    unique_clusters = dta_check_lvl['G_id'].unique()

    for cluster in unique_clusters:
        subset_dta = dta_check_lvl[dta_check_lvl['G_id'] == cluster]

        if len(subset_dta['S'].unique()) > 1 or len(subset_dta['D'].unique()) > 1 or len(subset_dta['Ng'].unique()) > 1:
            raise ValueError("Error: The values for S, D, and Ng must be consistent within each cluster (i.e., S, D, and Ng are cluster-level variables). Please verify that there are no discrepancies at the individual level within any cluster.")

# check_cluster_lvl(G_id, S, D, Ng)
# X.iloc[1, 0] = 0.3