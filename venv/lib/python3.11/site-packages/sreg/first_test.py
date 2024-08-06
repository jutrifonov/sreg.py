import sreg
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
res_data_gen=sreg_rgen(n=1000, tau_vec=[0, 10], cluster=True, Nmax=50, is_cov=False, n_strata = 5)

Y = res_data_gen["Y"]
S = res_data_gen["S"]
D = res_data_gen["D"]
N_g = res_data_gen["Ng"]
X = None
X = res_data_gen[['X1', 'X2']]
#X = pd.concat([X, N_g.rename('Ng')], axis=1)
G_id = res_data_gen["G_id"]
Ng = res_data_gen["Ng"]

result = sreg(Y = Y, S = S, D = D, G_id=G_id, Ng=Ng, X=None, HC1=True)
print(result)



# Function to add duplicate column with a temporary name and drop it after renaming
def add_duplicate_column(df, col_name):
    # Add duplicate column with a temporary name
    temp_col_name = f'{col_name}_duplicate'
    df[temp_col_name] = df[col_name]
    
    # Rename columns to have the same name as the original
    df.columns = [*df.columns[:-1], col_name]
    
    # Drop the temporary column
    df = df.drop(temp_col_name, axis=1)
    
    return df

# Function to handle duplicates
def handle_duplicates(df, X):
    # Check if 'Ng' is in X
    if 'Ng' in X.columns:
        # Rename the 'Ng' in X to a temporary name to avoid duplication issues
        X = X.rename(columns={'Ng': 'Ng_temp'})
    
    # Combine data for further calculations
    combined_df = pd.concat([df, X], axis=1)
    
    # If 'Ng_temp' is in the combined_df, rename it back to 'Ng'
    if 'Ng_temp' in combined_df.columns:
        combined_df = add_duplicate_column(combined_df, 'Ng')
    
    return combined_df
# Example usage
data = {
    'G_id': np.arange(1, 6),
    'Y_bar': np.random.randn(5),
    'Y': np.random.randn(5),
    'S': np.random.randint(1, 5, size=5),
    'D': np.random.randint(0, 3, size=5),
    'Ng': np.random.randint(10, 50, size=5)
}

df = pd.DataFrame(data)

# Ensure that Ng values in X_data are the same as in df
X_data = {
    'X1': np.random.randn(5),
    'X2': np.random.randn(5),
    'Ng': df['Ng']  # Duplicate Ng column in X with the same values
}

X = pd.DataFrame(X_data)

# Handle duplicates and combine data
result_df = handle_duplicates(df, X)

print(result_df)