import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\yourb\\OneDrive\\Documents\\NN\\NBA prediction\\NBA-NN-Prediction-\\full training set 2018-2022.csv'
df = pd.read_csv(file_path)

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Define the date range
start_date = pd.to_datetime('2018-10-16')
end_date = pd.to_datetime('2023-04-09')

# Filter the dataset
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'regular')]

# Create a column to identify games
df_filtered['game_id'] = df_filtered['date'].astype(str) + '_' + df_filtered['home'] + '_' + df_filtered['away']

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=df_filtered.columns)

# Process each game
for game_id, group in df_filtered.groupby('game_id'):
    home_team_row = group[group['team'] == group['home'].values[0]]
    away_team_row = group[group['team'] == group['away'].values[0]]
    
    if home_team_row.empty or away_team_row.empty:
        continue  # Skip if any team data is missing

    home_pts = home_team_row['PTS'].values[0]
    away_pts = away_team_row['PTS'].values[0]
    
    # Determine the winner
    if home_pts > away_pts:
        home_team_row['win'] = 1
        away_team_row['win'] = 0
    else:
        home_team_row['win'] = 0
        away_team_row['win'] = 1
    
    # Append results to the DataFrame
    results = pd.concat([results, home_team_row, away_team_row], ignore_index=True)

# Select relevant columns
columns_of_interest = ['date', 'home', 'away', 'team', 'PTS', 'FG%', '3P%', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'win']
results = results[columns_of_interest]

# Save or use the processed data
results.to_csv('filtered_dataset_with_team_win.csv', index=False)

# Display the first few rows of the filtered data
print(results.head())
