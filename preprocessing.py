import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'C:\\Users\\yourb\\OneDrive\\Documents\\NN\\NBA prediction\\NBA-NN-Prediction-\\full training set 2018-2022.csv'
df = pd.read_csv(file_path)
# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Define the date range
start_date = pd.to_datetime('10/16/2018')
end_date = pd.to_datetime('04/09/2023')

# Filter the dataset to include only regular season games and the specified date range
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'regular')]

# Drop the unnecessary columns
columns_to_drop = ['gameid', 'FTM', 'FTA', 'teamid', '3PM', '3PA', 'REB', 'TOV', 'PF','+/-', 'FGM', 'FGA', 'MIN']
df_filtered = df_filtered.drop(columns=columns_to_drop)

# Select relevant columns for normalization
columns_of_interest = ['PTS', 'FG%', '3P%', 'OREB', 'DREB', 'AST', 'STL', 'BLK','FT%']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the columns that need normalization
df_filtered[columns_of_interest] = scaler.fit_transform(df_filtered[columns_of_interest])

# Save or use the processed and normalized data
df_filtered.to_csv('new filtered_normalized_dataset_with_team_win.csv', index=False)

# Display the first few rows of the filtered and normalized data
print(df_filtered.head())
