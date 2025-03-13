import numpy as np
import mne
from tensorflow.keras.models import load_model

# ✅ EEG Processor Class
class EEGProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = None
        self.eeg_data = None
        self.labels = None

    def load_data(self):
        """Loads EEG data from the provided GDF file."""
        try:
            self.raw = mne.io.read_raw_gdf(self.file_path, preload=True)
            print(f"✅ Data loaded from {self.file_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def preprocess(self):
        """Preprocess the EEG data."""
        if self.raw:
            self.raw.pick_types(eeg=True)
            self.eeg_data = self.raw.get_data()

            # Normalize data
            self.eeg_data = (self.eeg_data - np.mean(self.eeg_data)) / np.std(self.eeg_data)
            print("✅ Data preprocessed successfully.")

            # Extract events and map using correct event IDs
            events, event_id = mne.events_from_annotations(self.raw)
            print(f"🔍 Detected Event IDs: {event_id}")

            # Map the event descriptions to their corresponding event IDs
            target_event_ids = [event_id[str(code)] for code in [769, 770, 771, 772] if str(code) in event_id]

            # Extract labels using the mapped IDs
            self.labels = [e[2] for e in events if e[2] in target_event_ids]
            print(f"✅ Extracted {len(self.labels)} labels.")

    def get_sample_signal(self, points=1000):
        """Return the first few points of EEG data for visualization."""
        return self.eeg_data[:, :points].tolist() if self.eeg_data is not None else []

    def get_data_and_labels(self):
        """Return preprocessed EEG data and corresponding labels."""
        return self.eeg_data, self.labels
