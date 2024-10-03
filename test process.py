import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'C:\\Users\\yourb\\OneDrive\\Documents\\NN\\NBA prediction\\NBA-NN-Prediction-\\full training set 2018-2022.csv'
df = pd.read_csv(file_path)

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Define the date range
start_date = pd.to_datetime('10/24/2023')
end_date = pd.to_datetime('04/14/2024')

# Filter the dataset to include only regular season games and the specified date range
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'regular')].copy()

# Create a 'win' column based on the maximum points scored in each game
df_filtered.loc[:, 'win'] = df_filtered.groupby(['home', 'away'])['PTS'].transform(lambda x: (x == x.max()).astype(int))

# Select only the relevant columns for normalization and prediction
df_relevant = df_filtered[['team', 'win']]

# Select columns for normalization
columns_of_interest = ['PTS', 'FG%', '3P%', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'FT%']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the columns that need normalization using .loc
df_filtered.loc[:, columns_of_interest] = scaler.fit_transform(df_filtered[columns_of_interest])

# Combine the relevant team and win data with normalized statistics
final_test_data = df_filtered[['team', 'win'] + columns_of_interest]

# Save the processed and normalized data to a CSV file
final_test_data.to_csv('testdata.csv', index=False)

print("Test data saved to 'testdata.csv'.")
