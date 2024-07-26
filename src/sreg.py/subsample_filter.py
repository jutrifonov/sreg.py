import pandas as pd
import numpy as np
#-------------------------------------------------------------------
# %#     Auxiliary function providing the appropriate data.frame
# %#     for the subsequent iterative OLS estimation. Takes into
# %#     account the number of observations and creates indicators.
#-------------------------------------------------------------------

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

# Display the first few rows of the dataframe
#print(data_clean.head())
#print(X.head())

def subsample_ols_sreg(Y, S, D, X, s, d):
    # Ensure s and d are lists
    if not isinstance(s, (list, tuple, np.ndarray)):
        s = [s]
    if not isinstance(d, (list, tuple, np.ndarray)):
        d = [d]
    
    # Convert X to a numpy array
    X = np.array(X)
    
    # Create a DataFrame
    data = pd.DataFrame({'Y': Y, 'S': S, 'D': D})
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, X.shape[1] + 1)])
    data = pd.concat([data, X_df], axis=1)
    
    # Filter data
    filtered_data = data[data['D'].isin(d) & data['S'].isin(s)]
    
    # Return the filtered data
    return filtered_data

#s = 3
#d = 0

#subsample_ols_sreg(Y, S, D, X, s, d)


def subsample_ols_creg(data, s, d):
    # Ensure s and d are scalar values (not lists)
    if isinstance(s, (list, tuple, np.ndarray)):
        s = s[0]
    if isinstance(d, (list, tuple, np.ndarray)):
        d = d[0]
    
    # Filter data
    filtered_data = data[(data['D'] == d) & (data['S'] == s)]
    
    # Return the filtered data
    return filtered_data

#data = pd.read_csv("/Users/trifonovjuri/Desktop/sreg.py/src/sreg.py/data_cl.csv")
#print(data.head())
#s = 1
#d = 1
#subsample_ols_creg(data, s, d)
