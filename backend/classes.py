import numpy as np
import mne
from tensorflow.keras.models import load_model

class EEGProcessor:
    """Processes EEG signals."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = None
        self.eeg_data = None

    def load_data(self):
        """Loads EEG GDF file."""
        try:
            print(f"🔄 Loading EEG file: {self.file_path} ...")
            self.raw = mne.io.read_raw_gdf(self.file_path, preload=True)
            self.eeg_data = self.raw.get_data(picks="eeg")  # Extract EEG data

            if self.eeg_data is None or self.eeg_data.size == 0:
                raise ValueError("❌ EEG data is empty. Please check the GDF file.")

            print(f"✅ EEG Data Loaded: {self.eeg_data.shape}")  # (channels, timepoints)
        except FileNotFoundError:
            print(f"❌ File not found: {self.file_path}. Please check the path.")
            self.eeg_data = None  # Prevent crashing
        except Exception as e:
            print(f"❌ Error loading EEG data: {e}")
            self.eeg_data = None

    def preprocess(self):
        """Preprocess EEG data."""
        if self.eeg_data is not None:
            self.eeg_data = (self.eeg_data - np.mean(self.eeg_data)) / np.std(self.eeg_data)
            print("✅ EEG data preprocessing complete.")
        else:
            print("❌ No EEG data available for preprocessing.")

class DeepLearningModel:
    """Loads trained deep learning model and makes predictions."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Loads the trained deep learning model."""
        try:
            print(f"🔄 Loading model from {self.model_path} ...")
            self.model = load_model(self.model_path)
            self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            print("✅ Model loaded successfully.")
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}. Please check the path.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def predict_movement(self, eeg_data):
        """Predict movement based on EEG data."""
        if self.model is None:
            print("❌ Model is not loaded. Please check your model path.")
            return None

        try:
            segment_length = 251
            n_segments = eeg_data.shape[1] // segment_length
            if n_segments == 0:
                raise ValueError("❌ EEG data too short for processing.")

            eeg_data_segments = np.array(np.split(eeg_data[:, :n_segments * segment_length], n_segments, axis=1))
            eeg_data_segments = eeg_data_segments.reshape(n_segments, 25, segment_length, 1)

            predictions = [np.argmax(self.model.predict(segment[np.newaxis, :, :, :])) for segment in eeg_data_segments]
            
            if not predictions:
                raise ValueError("❌ No valid predictions found.")

            final_prediction = np.clip(np.bincount(predictions).argmax(), 0, 2)  # Clip predictions

            movements = ["Jump", "Clap", "Stomp"]
            return movements[final_prediction]
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None

class GameEngine:
    """Controls the game based on EEG predictions."""
    def __init__(self):
        self.score = 0

    def update_game(self, movement):
        """Updates game state based on movement."""
        if movement == "Jump":
            self.score += 10
        elif movement == "Clap":
            self.score += 5
        elif movement == "Stomp":
            self.score += 8
        return f"Score: {self.score}"

# ✅ TEST CODE (Run this file directly to check outputs)
if __name__ == "__main__":
    print("🔄 Running EEGProcessor test...")

    # Simulate EEG file path (Ensure a valid GDF file exists in dataset/)
    eeg_file_path = "../dataset/BCICIV_2a_gdf/A01E.gdf" 

    # Test EEGProcessor
    eeg_processor = EEGProcessor(eeg_file_path)
    eeg_processor.load_data()
    eeg_processor.preprocess()

    # Test DeepLearningModel
    print("\n🔄 Running DeepLearningModel test...")
    model_path = "../backend/model/eeg_cnn_rnn_model.h5"  # Ensure the model exists
    model = DeepLearningModel(model_path)
    model.load_model()

    if eeg_processor.eeg_data is not None:
        prediction = model.predict_movement(eeg_processor.eeg_data)
        print(f"✅ Predicted Movement: {prediction}")
    else:
        print("❌ Skipping prediction due to missing EEG data.")

    # Test GameEngine
    print("\n🔄 Running GameEngine test...")
    game = GameEngine()
    if prediction:
        score = game.update_game(prediction)
        print(f"✅ GameEngine updated score: {score}")
    else:
        print("❌ No movement prediction available.")
