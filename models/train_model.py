# ✅ Import Required Libraries
import numpy as np
import mne  # For loading GDF files
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape

# ✅ Load and Preprocess EEG Data
print("🔄 Loading EEG dataset...")

# Load GDF file using MNE
data_path = "../BCICIV_2a_gdf/A01E.gdf"  # Ensure the correct path
raw = mne.io.read_raw_gdf(data_path, preload=True)

# Extract EEG signals & events
raw.filter(0.5, 40, fir_design='firwin')  # Apply band-pass filter
events, event_id = mne.events_from_annotations(raw)

# Handle repeated events by dropping duplicates
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=None, preload=True, event_repeated='drop')

# Convert EEG data to NumPy arrays
X = epochs.get_data()  # Shape: (trials, channels, timepoints)
y = epochs.events[:, -1]  # Get event labels

# Normalize EEG data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Reshape data for CNN input
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Convert labels to categorical
y = to_categorical(y)

# Split into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define CNN-RNN Model
print("🔄 Building CNN-RNN model...")
input_shape = (X_train.shape[1], X_train.shape[2], 1)
inputs = Input(shape=input_shape)

# CNN for spatial feature extraction
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Reshape for LSTM
x = Reshape((X_train.shape[1], -1))(x)

# LSTM for temporal feature extraction
x = LSTM(64, return_sequences=False)(x)

# Fully connected layers
x = Dense(32, activation='relu')(x)
outputs = Dense(y_train.shape[1], activation='softmax')(x)

# Compile Model
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Train the Model
print("🚀 Training CNN-RNN model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# ✅ Save the Model
model.save("../backend/model/eeg_cnn_rnn_model.h5")
print("✅ CNN-RNN Model saved successfully as 'eeg_cnn_rnn_model.h5'!")
