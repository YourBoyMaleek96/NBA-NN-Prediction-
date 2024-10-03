import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time

# Load the training dataset
train_file_path = 'C:\\Users\\MALIK.FREEMAN\\Desktop\\Malik SERN STUFF\\grad school\\OOM\\NBA-NN-Prediction-\\Trained_dataset.csv'
df_train = pd.read_csv(train_file_path)

# Prepare features and labels
features = ['PTS_home', 'FG%_home', '3P%_home', 'FT%_home', 
            'OREB_home', 'DREB_home', 'AST_home', 'STL_home', 'BLK_home', 
            'PTS_away', 'FG%_away', '3P%_away', 'FT%_away', 
            'OREB_away', 'DREB_away', 'AST_away', 'STL_away', 'BLK_away']

X_train = df_train[features]

# Convert winner to numerical values (0 and 1)
df_train['winner'] = df_train['winner'].apply(lambda x: 1 if x == 'DEN' else 0)  # Adjust this based on your team names
y_train = df_train['winner']

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Adjust output for binary classification
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
