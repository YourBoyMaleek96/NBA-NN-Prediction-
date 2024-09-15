import pandas as pd

# Load the dataset
file_path = r'C:\Users\yourb\OneDrive\Documents\NN\NBA prediction\NBA-NN-Prediction-\full training set 2018-2022.csv'
df = pd.read_csv(file_path)
# Filter for regular season games
df_regular_season = df[df['season_type'] == 'Regular Season']
# Convert the 'game_date' column to datetime
df_regular_season['game_date'] = pd.to_datetime(df_regular_season['game_date'], format='%m/%d/%Y %H:%M')

# Define the date range
start_date = '2018-10-16'
end_date = '2023-04-09'

# Filter the dataframe by date range
df_filtered = df_regular_season[(df_regular_season['game_date'] >= start_date) & (df_regular_season['game_date'] <= end_date)]
df_filtered.to_csv(r'C:\Users\yourb\OneDrive\Documents\NN\NBA prediction\NBA-NN-Prediction-\filtered_training_set.csv', index=False)
