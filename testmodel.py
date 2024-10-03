import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib  # Import joblib for saving the scaler


# Step 7: Load the test data
test_file_path = r'C:\Users\yourb\OneDrive\Documents\NN\NBA prediction\NBA-NN-Prediction-\testdata.csv'
test_df = pd.read_csv(test_file_path)

# Step 8: Normalize the test data using the saved scaler
test_df[features] = joblib.load('scaler.pkl').transform(test_df[features])

# Step 9: Prepare the input features
X_test = test_df[features]

# Step 10: Load the saved model
loaded_model = keras.models.load_model('nba_prediction_model.h5')

# Step 11: Make predictions
predictions = loaded_model.predict(X_test)
predicted_wins = (predictions > 0.5).astype(int)  # Convert probabilities to binary

# Step 12: Add predictions to the DataFrame
test_df['predicted_win'] = predicted_wins

# Step 13: Evaluate predictions (if actual results are available)
if 'win' in test_df.columns:
    accuracy = (test_df['predicted_win'].values.flatten() == test_df['win']).mean()
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the results with predictions
test_df.to_csv('predicted_results.csv', index=False)
print("Predictions saved to 'predicted_results.csv'.")
