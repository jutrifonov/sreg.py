# data for testing the development
import pandas as pd
import numpy as np
from sreg import sreg
from sreg import sreg_rgen


# Load the dataframe from the CSV file
data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg.py/src/sreg.py/data.csv")

# Display the first few rows of the dataframe
print(data.head())

# Select the columns
Y = data['gradesq34']
D = data['treatment']
S = data['class_level']

# Create a new DataFrame with selected columns
data_clean = pd.DataFrame({'Y': Y, 'D': D, 'S': S})

# Replace values in column D
data_clean['D'] = data_clean['D'].apply(lambda x: 0 if x == 3 else x)

# Extract the columns again
Y = data_clean['Y']
D = data_clean['D']
S = data_clean['S']

# Create a contingency table
contingency_table = pd.crosstab(data_clean['D'], data_clean['S'])
print(contingency_table)

# Select the columns
Y = data['gradesq34']
D = data['treatment']
S = data['class_level']
pills = data['pills_taken']
age = data['age_months']

# Create a new DataFrame with selected columns
data_clean = pd.DataFrame({'Y': Y, 'D': D, 'S': S, 'pills': pills, 'age': age})

# Replace values in column D
data_clean['D'] = data_clean['D'].apply(lambda x: 0 if x == 3 else x)

# Extract the columns again
Y = data_clean['Y']
D = data_clean['D']
S = data_clean['S']
X = data_clean[['pills', 'age']]

# Display the first few rows of the dataframe
print(data_clean.head())
print(X.head())




data=sreg_rgen(n=1000, tau_vec=[0, 0.2], n_strata=4, cluster=True, is_cov = True)
Y = data["Y"]
S = data["S"]
D = data["D"]
X = data[['X1', 'X2']]
G_id = data["G_id"]
Ng = data["Ng"]
X.iloc[2, 0] = 1.34
result = sreg(Y=Y, S=S, D=D, G_id=G_id, Ng=Ng, X=X)
print(result)
