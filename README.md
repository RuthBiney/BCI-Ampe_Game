Hybrid CNN-RNN for EEG-Based Ampe Game
Overview
This project develops a Brain-Computer Interface (BCI) system for motor imagery classification using a hybrid CNN-RNN model. The system is integrated with the traditional Ghanaian game Ampe, enabling users to control the game using their brain signals. The project leverages publicly available EEG datasets, deep learning frameworks, and open-source tools to create a low-cost and accessible solution for motor imagery classification.

Features
Hybrid CNN-RNN Model: Combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) for temporal dependencies in EEG signals.

Ampe Game Integration: Allows users to control the traditional Ghanaian game Ampe using motor imagery (e.g., imagining hand claps or foot stomps).

Real-Time Classification: Classifies EEG signals in real-time with low latency.

User-Friendly Interface: Built using Streamlit for an interactive and intuitive user experience.

Open-Source: All code and documentation are publicly available for further research and development.

Installation
To set up the project locally, follow these steps:

1. Clone the Repository
   git https://github.com/RuthBiney/BCI-Ampe_Game.git

2. Install Dependencies
   Install the required Python libraries using pip:
   pip install -r requirements.txt

3. Download the Dataset
<!-- Have to replace with the Actual Ampe Dataset I'm working on.  -->

Usage

1. Preprocess the EEG Data
   Run the preprocessing script to clean and prepare the EEG data:
   python preprocess.py

This script performs the following steps:
Filters the raw EEG signals (e.g., bandpass filter between 7–30 Hz).
Normalizes the data.
Segments the data into time windows (e.g., 1-second windows at 128 Hz).

2. Train the Model
   Train the hybrid CNN-RNN model using the preprocessed data:
   python train.py

This script:
Loads the preprocessed data.
Defines and compiles the hybrid CNN-RNN model.
Trains the model and saves it to models/hybrid_cnn_rnn_model.h5.

3. Run the Streamlit App
   Start the Streamlit app to interact with the Ampe game:
   streamlit run ampe_game.py

The app will open in your browser, allowing you to:
Simulate EEG data input.
Predict motor imagery tasks (e.g., left hand, right hand, foot).
Control the Ampe game using the predicted actions.

Model Architecture
The hybrid CNN-RNN model consists of the following layers:
Input Layer:
Reshapes the EEG data for CNN input.
CNN Layers:
Conv2D (32 filters, 3x3 kernel, ReLU activation).
MaxPooling2D (2x2 window).
Flatten.
RNN Layers:
LSTM (64 units, return sequences).
LSTM (32 units).
Output Layer:
Dense (4 units, softmax activation for 4 classes).

Testing with Local Users

<!-- To test the system with real users:

Connect an EEG headset (e.g., OpenBCI Ganglion or Emotiv Epoc).

Run the Streamlit app and guide users through the process of playing the Ampe game using their brain signals.

Collect feedback and refine the system based on user input. -->  We Might not be using the EEG headset for the testing.

Deployment
Local Deployment
Run the Streamlit app locally:
streamlit run ampe_game.py

Cloud Deployment
Deploy the app on a cloud platform like Streamlit Sharing or Heroku:

Streamlit Sharing:

Upload the code to GitHub and connect the repository to Streamlit Sharing.

<!-- Contributing
Contributions are welcome! To contribute to this project:

Fork the repository.

Create a new branch for your feature or bug fix.

Submit a pull request with a detailed description of your changes. -->

License
This project is licensed under the MIT License.

Acknowledgments

TensorFlow and Keras for deep learning model development.

Streamlit for building the user interface.

MNE-Python for EEG signal processing.
