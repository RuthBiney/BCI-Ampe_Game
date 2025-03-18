import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import mne
from tensorflow.keras.models import load_model
import os

st.title("🧠 Real-Time EEG Ampe Classification")

uploaded_file = st.file_uploader("Upload EEG GDF", type=["gdf"])

if uploaded_file is not None:
    temp_file_path = "temp.gdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        raw = mne.io.read_raw_gdf(temp_file_path, preload=True)
        st.success("✅ EEG file loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading EEG file: {e}")
        st.stop()

    eeg_data = raw.get_data(picks="eeg")
    if eeg_data.shape[0] > 25:
        eeg_data = eeg_data[:25, :]

    df_subset = pd.DataFrame(eeg_data[:, :1000].T, columns=raw.ch_names[:eeg_data.shape[0]])
    fig = px.line(df_subset, title="EEG Signal (First 1000 Time Points)")
    st.plotly_chart(fig)

    model_path = "../backend/model/eeg_transformer_model.keras"
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        st.error("❌ Model file not found. Ensure it's at '../backend/model/eeg_transformer_model.keras'.")
        st.stop()

    segment_length = 1000
    n_segments = eeg_data.shape[1] // segment_length
    if n_segments == 0:
        st.error("❌ Not enough EEG data for processing.")
        st.stop()

    eeg_data_segments = np.array(np.split(eeg_data[:, :n_segments * segment_length], n_segments, axis=1))
    eeg_data_segments = eeg_data_segments.reshape(n_segments, 25, segment_length, 1)

    predictions = []
    for segment in eeg_data_segments:
        segment = segment[np.newaxis, :, :, :]
        pred = model.predict(segment)
        predictions.append(np.argmax(pred))

        st.write(f"Raw prediction scores for segment: {pred}")

    if len(predictions) == 0:
        st.error("❌ No valid predictions found.")
        st.stop()

    final_prediction = np.bincount(predictions).argmax()
    movements = ["Left Hand Movement", "Right Hand Movement", "Both Feet Movement", "Tongue Movement"]

    if final_prediction >= len(movements):
        st.error(f"❌ Invalid prediction index: {final_prediction}.")
    else:
        st.subheader(f"🧠 Predicted Movement: {movements[final_prediction]}")

    st.success("✅ EEG Processing Complete!")

    # Clean up temporary file
    os.remove(temp_file_path)
