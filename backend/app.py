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
            raw = mne.io.read_raw_gdf(file_path, preload=True)
            eeg_data = raw.get_data(picks="eeg")  # Shape: (n_channels, n_times)
            if eeg_data.shape[0] > 25:  # Ensure we only keep the first 25 channels
                eeg_data = eeg_data[:25, :]

            # ✅ Preprocess EEG for Model
            segment_length = 1000  # Time points per segment (match model input)
            n_segments = eeg_data.shape[1] // segment_length  # Compute number of segments

            # Ensure we have at least one valid segment
            if n_segments == 0:
                return jsonify({"error": f"Not enough EEG data for processing in {filename}."}), 400

            # Reshape into (n_segments, 25, 1000, 1)
            eeg_data_segments = np.array(np.split(eeg_data[:, :n_segments * segment_length], n_segments, axis=1))
            eeg_data_segments = eeg_data_segments.reshape(n_segments, 25, segment_length, 1)

            # ✅ Predict Movements
            predictions = []
            for segment in eeg_data_segments:
                segment = segment[np.newaxis, :, :, :]  # Add batch dimension
                pred = model.predict(segment)
                predictions.append(np.argmax(pred))

                # Debugging: Print raw prediction scores
                print(f"Raw prediction scores for segment: {pred}")

            # ✅ Ensure Predictions Are Valid
            if len(predictions) == 0:
                return jsonify({"error": f"No valid predictions found in {filename}."}), 400

            # ✅ Handle Invalid Predictions
            try:
                final_prediction = int(np.bincount(predictions).argmax())  # Majority voting
                movement_type = movement_types.get(final_prediction, "Unknown Movement")

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
                    "predicted_class": final_prediction,
                    "confidence_score": float(round(pred.mean(), 4)),  # Convert to standard Python float
                    "movement_type": movement_type,
                    "signal_visualization": f"data:image/png;base64,{img_base64}"  # Add data URI scheme
                })

                os.remove(file_path)  # Clean up temporary file

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"Results: {results}")  # Debugging statement to check the results
    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)
