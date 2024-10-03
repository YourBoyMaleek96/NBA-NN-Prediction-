import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\full training set.csv'
df = pd.read_csv(file_path)

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Define the date range
start_date = pd.to_datetime('10/16/2018')
end_date = pd.to_datetime('04/09/2023')

# Filter the dataset to include only regular season games and the specified date range
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'regular')].copy()

# Initialize an empty list to store the new data
new_data = []

# Iterate over the filtered data in steps of 2 to pair the teams that played in each game
for i in range(0, len(df_filtered), 2):
    row_home = df_filtered.iloc[i]
    row_away = df_filtered.iloc[i + 1]

    # Gather statistics for the home team
    home_stats = [
        row_home['team'],  # home_team
        row_away['team'],  # away_team
        row_home['PTS'],   # PTS_home
        row_home['FG%'],   # FG%_home
        row_home['3P%'],   # 3P%_home
        row_home['FT%'],   # FT%_home
        row_home['OREB'],  # OREB_home
        row_home['DREB'],  # DREB_home
        row_home['AST'],   # AST_home
        row_home['STL'],   # STL_home
        row_home['BLK']    # BLK_home
    ]

    # Gather statistics for the away team
    away_stats = [
        row_away['PTS'],   # PTS_away
        row_away['FG%'],   # FG%_away
        row_away['3P%'],   # 3P%_away
        row_away['FT%'],   # FT%_away
        row_away['OREB'],  # OREB_away
        row_away['DREB'],  # DREB_away
        row_away['AST'],   # AST_away
        row_away['STL'],   # STL_away
        row_away['BLK']    # BLK_away
    ]

    # Determine the winner
    winner = row_home['team'] if row_home['win'] == 1 else row_away['team']

    # Combine home and away stats and the winner into a single list
    new_data.append(home_stats + away_stats + [winner])

# Create a DataFrame from the new data
columns = ['home_team', 'away_team', 'PTS_home', 'FG%_home', '3P%_home', 'FT%_home', 
           'OREB_home', 'DREB_home', 'AST_home', 'STL_home', 'BLK_home', 
           'PTS_away', 'FG%_away', '3P%_away', 'FT%_away', 'OREB_away', 'DREB_away', 
           'AST_away', 'STL_away', 'BLK_away', 'winner']

df_new = pd.DataFrame(new_data, columns=columns)

# Normalize relevant columns using Min-Max scaling
stats_columns = ['PTS_home', 'FG%_home', '3P%_home', 'FT%_home', 
                 'OREB_home', 'DREB_home', 'AST_home', 'STL_home', 'BLK_home',
                 'PTS_away', 'FG%_away', '3P%_away', 'FT%_away', 
                 'OREB_away', 'DREB_away', 'AST_away', 'STL_away', 'BLK_away']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the columns that need normalization
df_new[stats_columns] = scaler.fit_transform(df_new[stats_columns])

# Save the new DataFrame to a CSV file
df_new.to_csv('Trained_dataset_normalized.csv', index=False)

print("Processed and normalized data saved to 'Trained_dataset_normalized.csv'.")
