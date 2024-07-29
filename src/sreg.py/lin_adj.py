import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
#-------------------------------------------------------------------
# %#     Function that implements the calculation of \hat{\mu} --
# %#     i.e., calculates linear adjustments
# Load the dataframe from the CSV file
#data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg.py/src/sreg.py/data.csv")

# Display the first few rows of the dataframe
#print(data.head())

# Select the columns
#Y = data['gradesq34']
#D = data['treatment']
#S = data['class_level']

# Create a new DataFrame with selected columns
#data_clean = pd.DataFrame({'Y': Y, 'D': D, 'S': S})

# Replace values in column D
#data_clean['D'] = data_clean['D'].apply(lambda x: 0 if x == 3 else x)

# Extract the columns again
#Y = data_clean['Y']
#D = data_clean['D']
#S = data_clean['S']

# Create a contingency table
#contingency_table = pd.crosstab(data_clean['D'], data_clean['S'])
#print(contingency_table)

# Select the columns
#Y = data['gradesq34']
#D = data['treatment']
#S = data['class_level']
#pills = data['pills_taken']
#age = data['age_months']

# Create a new DataFrame with selected columns
#data_clean = pd.DataFrame({'Y': Y, 'D': D, 'S': S, 'pills': pills, 'age': age})

# Replace values in column D
#data_clean['D'] = data_clean['D'].apply(lambda x: 0 if x == 3 else x)

# Extract the columns again
#Y = data_clean['Y']
#D = data_clean['D']
#S = data_clean['S']
#X = data_clean[['pills', 'age']]


#model = lm_iter_sreg(Y, S, D, X)

def lin_adj_sreg(a, S, X, model):
    # Combine S and X into a DataFrame
    data = pd.DataFrame({'S': S})
    X_df = pd.DataFrame(X)
    data = pd.concat([data, X_df], axis=1)

    # Extract the theta matrix from the model
    theta_mtrx = model[a]

    # Match theta vectors to the S values
    theta_vec_mapped = theta_mtrx[S-1, :]

    # Calculate mu_hat
    mu_hat = np.einsum('ij,ij->i', X, theta_vec_mapped)
    
    return mu_hat

#a = 0
#lin_adj_sreg(a, S, X, model)


#data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg.py/src/sreg.py/data_cl.csv")
#print(data.head())
#Y = data['Y']
#D = data['D']
#S = data['S']
#G_id = data['G.id']
#Ng = data['Ng']
#X = data[['x_1', 'x_2']]

model = lm_iter_creg(Y, S, D, G_id, Ng, X)

def lin_adj_creg(a, data, model):
    # Extract the X.data part of the data
    X_data = data.iloc[:, 6:]  # Select columns from the 6th to the last
    
    # Extract the theta matrix from the model
    theta_mtrx = model['theta_list'][a]
    
    # Match theta vectors to the S values
    theta_vec_mapped = theta_mtrx[data['S'].values - 1, :]
    
    # Calculate mu.hat
    mu_hat = np.einsum('ij,ij->i', X_data.values, theta_vec_mapped)
    
    return mu_hat

#a = 0
data = model['cl_lvl_data']
#lin_adj_creg(a, data, model)