# data for testing the development
import pandas as pd
import numpy as np


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