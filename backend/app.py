from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import io
import base64
from classes import EEGProcessor

app = Flask(__name__)

# ✅ Increase Flask's File Upload Limit to 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

# ✅ Load Transformer Model
MODEL_PATH = "model/eeg_transformer_model.keras"
print(f"🔄 Loading Transformer model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Movement type mapping
movement_types = {
    0: "Left Hand Movement",
    1: "Right Hand Movement",
    2: "Both Feet Movement",
    3: "Tongue Movement"
}

# ✅ Home Route to Avoid 404 Errors
@app.route('/')
def home():
    return "🚀 EEG Transformer API is Running! Use /predict to make predictions."

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded."}), 400

    results = []

    for file in uploaded_files:
        filename = file.filename
        file_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        file.save(file_path)

        try:
            # ✅ Process EEG Data
            processor = EEGProcessor(file_path)
            processor.load_data()
            processor.preprocess()

            eeg_data, labels = processor.get_data_and_labels()
            if eeg_data is None or not labels:
                return jsonify({"error": f"No valid EEG data found in {filename}."}), 400

            # ✅ Preprocess EEG for Model
            eeg_data = mne.filter.filter_data(eeg_data, sfreq=processor.raw.info['sfreq'], l_freq=8, h_freq=30)
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

            segment_length = 1000
            n_channels, n_times = eeg_data.shape
            n_segments = n_times // segment_length
            eeg_data = eeg_data[:, :n_segments * segment_length].reshape(n_channels, n_segments, segment_length)
            eeg_data = eeg_data.transpose(1, 0, 2)[..., np.newaxis]

            # Debugging: Print shape of EEG data
            print(f"EEG data shape for {filename}: {eeg_data.shape}")

            # ✅ Make Predictions
            preds = model.predict(eeg_data)
            pred_scores = preds.mean(axis=0)
            predicted_class = int(np.argmax(pred_scores))
            confidence_score = float(pred_scores[predicted_class])
            movement_type = movement_types.get(predicted_class, "Unknown Movement")

            # Debugging: Print prediction scores
            print(f"Prediction scores for {filename}: {pred_scores}")
            print(f"Predicted class for {filename}: {predicted_class}")
            print(f"Confidence score for {filename}: {confidence_score}")
            print(f"Movement type for {filename}: {movement_type}")

            # ✅ Generate Signal Visualization
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(eeg_data[0].squeeze())
            ax.set_title(f"EEG Signal for {filename}")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            results.append({
                "filename": filename,
                "predicted_class": predicted_class,
                "confidence_score": round(confidence_score, 4),
                "movement_type": movement_type,
                "signal_visualization": img_base64
            })

            os.remove(file_path)  # Clean up temporary file

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"Results: {results}")  # Debugging statement to check the results
    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)
