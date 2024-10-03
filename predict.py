import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Load the dataset with the test data
test_file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\fulltestdata.csv'
test_data = pd.read_csv(test_file_path)

# Load the pre-trained model
model_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\nba_prediction_model.h5'
model = load_model(model_path)

# Prepare the features for prediction (remove 'winner' and other non-stat columns)
features = test_data[['PTS_home', 'FG%_home', '3P%_home', 'FT%_home', 'OREB_home', 'DREB_home', 
                      'AST_home', 'STL_home', 'BLK_home', 'PTS_away', 'FG%_away', '3P%_away', 
                      'FT%_away', 'OREB_away', 'DREB_away', 'AST_away', 'STL_away', 'BLK_away']]

# Predict the game outcomes (probabilities of winning)
predictions = model.predict(features)

# Convert probabilities to binary predictions (0 for away team win, 1 for home team win)
predicted_winners = np.where(predictions >= 0.5, 1, 0)

# Map the predictions back to team names (home team wins if prediction is 1, else away team wins)
test_data['predicted_winner'] = np.where(predicted_winners == 1, test_data['home_team'], test_data['away_team'])

# Save the predictions to a new CSV file
predicted_file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\Grad school\\OOM\\NBA-NN-Prediction-\\predict.csv'
test_data[['home_team', 'away_team', 'predicted_winner']].to_csv(predicted_file_path, index=False)

# Calculate the accuracy by comparing actual winners to predicted winners
accuracy = accuracy_score(test_data['winner'], test_data['predicted_winner'])
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Optionally save the accuracy result
print(f"Predictions saved to '{predicted_file_path}'")
