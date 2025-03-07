# ✅ Import Required Libraries
import numpy as np
import mne  # For loading GDF files
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D

# ✅ Load and Preprocess EEG Data
print("🔄 Loading EEG dataset...")

# Load GDF file using MNE
data_path = "../BCICIV_2a_gdf/A01E.gdf"  
raw = mne.io.read_raw_gdf(data_path, preload=True)

# Extract EEG signals & events
raw.filter(0.5, 40, fir_design='firwin')  
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

# ✅ Define Transformer Block
def transformer_block(inputs, num_heads=4, key_dim=64):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = Add()([inputs, attn_output])  # Skip connection
    attn_output = LayerNormalization()(attn_output)
    return attn_output

# ✅ Define Transformer-Based Model
print("🔄 Building Transformer model...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Reshape((X_train.shape[1], -1))(x)

# Apply Transformer
x = transformer_block(x)

# Flatten the output to remove the time dimension
x = GlobalAveragePooling1D()(x)

# Fully connected layers
x = Dense(128, activation='relu')(x)
outputs = Dense(y_train.shape[1], activation='softmax')(x)

# Compile Model
model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ✅ Train the Model
print("🚀 Training Transformer model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# ✅ Save the Model
model.save("../backend/model/eeg_transformer_model.h5")
print("✅ Transformer Model saved successfully as 'eeg_transformer_model.h5'!")
