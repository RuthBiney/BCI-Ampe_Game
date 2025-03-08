from flask import Flask, request, jsonify
from flask_cors import CORS
from classes import EEGProcessor, DeepLearningModel, GameEngine

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow frontend access

# ✅ Load Model & Game Engine
model_path = "../backend/model/eeg_cnn_rnn_model.h5"  
model = DeepLearningModel(model_path)
model.load_model()

game = GameEngine()

@app.route('/')
def home():
    """Home route to check if the API is running."""
    return jsonify({"message": "EEG Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    """API to classify EEG movement & update game score."""
    try:
        # ✅ Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No EEG file uploaded"}), 400
        
        file = request.files['file']
        file_path = "temp.gdf"
        file.save(file_path)  # Save EEG file temporarily
        
        # ✅ Process EEG Data
        eeg_processor = EEGProcessor(file_path)
        eeg_processor.load_data()
        eeg_processor.preprocess()

        if eeg_processor.eeg_data is None:
            return jsonify({"error": "Invalid EEG data"}), 400

        # ✅ Predict Movement
        movement = model.predict_movement(eeg_processor.eeg_data)

        if movement is None:
            return jsonify({"error": "Prediction failed"}), 500

        # ✅ Update Game Score
        score = game.update_game(movement)

        return jsonify({"movement": movement, "score": score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
