import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import mne
from tensorflow.keras.models import load_model

# ✅ Streamlit UI
st.title("🧠 Real-Time EEG Ampe Classification")

# ✅ Upload EEG GDF file
uploaded_file = st.file_uploader("Upload EEG GDF", type=["gdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.gdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Load EEG Data from GDF File
    try:
        raw = mne.io.read_raw_gdf("temp.gdf", preload=True)
        st.success("✅ EEG file loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading EEG file: {e}")
        st.stop()

    # Extract EEG data (select first 25 channels)
    eeg_data = raw.get_data(picks="eeg")  # Shape: (n_channels, n_times)
    if eeg_data.shape[0] > 25:  # Ensure we only keep the first 25 channels
        eeg_data = eeg_data[:25, :]

    # ✅ Visualize EEG Signal (Only First 1000 Time Points)
    df_subset = pd.DataFrame(eeg_data[:, :1000].T, columns=raw.ch_names[:eeg_data.shape[0]])
    fig = px.line(df_subset, title="EEG Signal (First 1000 Time Points)")
    st.plotly_chart(fig)

    # ✅ Load Transformer Model & Compile
    model_path = "../backend/model/eeg_transformer_model.keras"
    try:
        model = load_model(model_path)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # ✅ Fix: Compile Model
    except FileNotFoundError:
        st.error("❌ Model file not found. Ensure it's at '../backend/model/eeg_transformer_model.keras'.")
        st.stop()

    # ✅ Preprocess EEG Data for Model
    segment_length = 1000  # Time points per segment (match model input)
    n_segments = eeg_data.shape[1] // segment_length  # Compute number of segments

    # Ensure we have at least one valid segment
    if n_segments == 0:
        st.error("❌ Not enough EEG data for processing. Ensure the recording is long enough.")
        st.stop()

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
        st.write(f"Raw prediction scores for segment: {pred}")

    # ✅ Ensure Predictions Are Valid
    if len(predictions) == 0:
        st.error("❌ No valid predictions found. Check EEG data format.")
        st.stop()

    # ✅ Handle Invalid Predictions
    try:
        final_prediction = np.bincount(predictions).argmax()  # Majority voting
        movements = ["Left Hand Movement", "Right Hand Movement", "Both Feet Movement", "Tongue Movement"]
        
        if final_prediction >= len(movements):  # Avoid index out of range
            st.error(f"❌ Invalid prediction index: {final_prediction}.")
        else:
            st.subheader(f"🧠 Predicted Movement: {movements[final_prediction]}")
    except ValueError:
        st.error("❌ Unable to determine prediction. Please retry with valid EEG data.")

    st.success("✅ EEG Processing Complete!")
    