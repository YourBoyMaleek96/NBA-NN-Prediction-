import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\full training set.csv'
df = pd.read_csv(file_path)

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Define the date range
start_date = pd.to_datetime('10/24/2023')
end_date = pd.to_datetime('04/14/2024')

# Filter the dataset to include only regular season games and the specified date range
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'regular')].copy()

# Initialize an empty list to store matchup results along with statistics
matchups = []

# Iterate over the filtered data in steps of 2 to pair the teams that played in each game
for i in range(0, len(df_filtered), 2):
    row_home = df_filtered.iloc[i]
    row_away = df_filtered.iloc[i + 1]

    # Determine the winning team based on the 'win' column
    if row_home['win'] == 1:
        winner = row_home['team']
    else:
        winner = row_away['team']

    # Store the matchup, the winning team, and the stats
    matchups.append({
        'home_team': row_home['team'],
        'away_team': row_away['team'],
        'PTS_home': row_home['PTS'],
        'FG%_home': row_home['FG%'],
        '3P%_home': row_home['3P%'],
        'FT%_home': row_home['FT%'],
        'OREB_home': row_home['OREB'],
        'DREB_home': row_home['DREB'],
        'AST_home': row_home['AST'],
        'STL_home': row_home['STL'],
        'BLK_home': row_home['BLK'],
        'PTS_away': row_away['PTS'],
        'FG%_away': row_away['FG%'],
        '3P%_away': row_away['3P%'],
        'FT%_away': row_away['FT%'],
        'OREB_away': row_away['OREB'],
        'DREB_away': row_away['DREB'],
        'AST_away': row_away['AST'],
        'STL_away': row_away['STL'],
        'BLK_away': row_away['BLK'],
        'winner': winner
    })

# Create a DataFrame for the matchups with statistics
df_matchups = pd.DataFrame(matchups)

# Save the matchups with statistics to a CSV file
output_file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\fulltestdata.csv'
df_matchups.to_csv(output_file_path, index=False)

print(f"Matchup data with statistics and winners saved to '{output_file_path}'.")
