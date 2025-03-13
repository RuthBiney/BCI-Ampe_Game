import os
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample, class_weight
from sklearn.model_selection import train_test_split
import mne

# ✅ Import custom transformer model
from transformer_model import build_transformer_model

# ✅ Add backend to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes import EEGProcessor

# Dataset Directory
DATASET_DIR = "../dataset/BCICIV_2a_gdf"

# Lists to store data and labels
all_data = []
all_labels = []

# ✅ Data Augmentation Functions
def add_noise(data, noise_factor=0.01):
    return data + noise_factor * np.random.randn(*data.shape)

def time_shift(data, shift_max=100):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=1)

def random_crop(data, crop_length=1000):
    if data.shape[1] > crop_length:
        start = np.random.randint(0, data.shape[1] - crop_length)
        return data[:, start:start + crop_length]
    else:
        return data

# ✅ Process Each GDF File
for filename in os.listdir(DATASET_DIR):
    if filename.endswith('.gdf'):
        print(f"🔄 Processing {filename}...")
        processor = EEGProcessor(os.path.join(DATASET_DIR, filename))
        processor.load_data()
        processor.preprocess()

        eeg_data, labels = processor.get_data_and_labels()

        if eeg_data is not None and labels:
            eeg_data = mne.filter.filter_data(eeg_data, sfreq=processor.raw.info['sfreq'], l_freq=8, h_freq=30)
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

            n_channels, n_times = eeg_data.shape
            segment_length = 1000
            n_segments = min(n_times // segment_length, len(labels))

            eeg_data = eeg_data[:, :n_segments * segment_length]
            eeg_data = eeg_data.reshape(n_channels, n_segments, segment_length).transpose(1, 0, 2)

            for segment, label in zip(eeg_data, labels[:n_segments]):
                if label in [7, 8, 9, 10]:
                    all_data.append(segment)
                    all_labels.append(label)

                    # Augment and append
                    all_data.append(add_noise(segment))
                    all_labels.append(label)
                    all_data.append(time_shift(segment))
                    all_labels.append(label)
                    all_data.append(random_crop(segment))
                    all_labels.append(label)

# ✅ Convert to NumPy Arrays
X = np.array(all_data)
y = np.array(all_labels)

# ✅ Balance the Dataset
classes = [7, 8, 9, 10]
min_samples = min([list(y).count(c) for c in classes])
X_balanced, y_balanced = [], []

for c in classes:
    idx = np.where(y == c)[0]
    X_c, y_c = X[idx], y[idx]
    X_resampled, y_resampled = resample(X_c, y_c, replace=False, n_samples=min_samples, random_state=42)
    X_balanced.append(X_resampled)
    y_balanced.append(y_resampled)

X_balanced = np.concatenate(X_balanced)[..., np.newaxis]
y_balanced = np.concatenate(y_balanced)
y_balanced = np.array([0 if label == 7 else 1 if label == 8 else 2 if label == 9 else 3 for label in y_balanced])
y_balanced = to_categorical(y_balanced, 4)

print(f"✅ Final Dataset Shape: {X_balanced.shape}")
print(f"✅ Label Distribution: {np.bincount(y_balanced.argmax(axis=1))}")

# ✅ Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced.argmax(axis=1), random_state=42
)

# ✅ Build Transformer Model
model = build_transformer_model((X_balanced.shape[1], X_balanced.shape[2], 1))
optimizer = Adam(learning_rate=0.0001)
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
print("✅ Transformer Model compiled successfully.")

# ✅ Class Weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_balanced.argmax(axis=1)), y=y_balanced.argmax(axis=1))
class_weights = dict(enumerate(class_weights))

# ✅ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ✅ Train Model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val),
          callbacks=[early_stopping, reduce_lr], class_weight=class_weights)

# ✅ Save Model in `.keras` Format
model.save("../model/eeg_transformer_model.keras")
print("✅ Transformer model trained and saved successfully.")
