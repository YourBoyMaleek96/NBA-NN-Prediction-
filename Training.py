import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time

# Load the training dataset
train_file_path = 'C:\\Users\\yourb\\OneDrive\\Documents\\NN\\NBA prediction\\NBA-NN-Prediction-\\new filtered_normalized_dataset_with_team_win.csv'

df_train = pd.read_csv(train_file_path)

# Load the test dataset
test_file_path = 'test data.csv'
df_test = pd.read_csv(test_file_path)

# Prepare features and labels
features = ['PTS', 'FG%', '3P%', 'OREB', 'DREB', 'AST', 'STL', 'BLK']
X_train = df_train[features]
y_train = df_train['win']
X_test = df_test[features]
y_test = df_test['win']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model and measure time
start_time = time.time()
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Print the final accuracy
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

# Save the model
model.save('nba_prediction_model.h5')
print("Model saved as 'nba_prediction_model.h5'")
