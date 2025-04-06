**Brain-Computer Interface for Motor Imagery**
This project is a Brain-Computer Interface (BCI) system designed to classify EEG signals captured during motor imagery tasks. Using deep learning techniques, the system interprets user brainwave data and predicts the type of imagined movement in near real-time. The goal is to facilitate intuitive human-computer interaction for applications in assistive technology, neuroscience, and neurofeedback systems.

**Features**
🔐 User registration and login with input validation

📤 EEG data upload for classification (.gdf files from BCI Competition IV-2a)

⚙️ Real-time processing and prediction of motor imagery tasks

📈 Display of classification confidence and visual output

🖥️ Responsive and intuitive web frontend

🤖 Deep learning model trained on multi-class EEG signals

✅ Multiple EEG file upload support

🧪 Testing and performance evaluation (Precision: 0.8289 | Recall: 0.8291 | F1-score: 0.8287)

BCI-Motor-Imagery/
│
├── backend/ # Python backend with model, APIs, preprocessing
│ ├── model/ # Trained transformer-based model
│ ├── datasetfolder/ # EEG .gdf files
│ ├── preprocessing/ # Signal processing scripts
│ └── app.py # Flask or FastAPI app for API endpoints
│
├── frontend/ # HTML, CSS, JavaScript frontend
│ ├── index.html # Landing page
│ ├── signup.html # Sign up page
│ ├── login.html # Login page
│ ├── data_input.html # EEG data upload page
│ └── result.html # Prediction output display
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

**Tech Stack**
Frontend:

Typescript, React

Animations and transitions for user interaction

Backend:

Python 3.x

Flask or FastAPI (for API endpoints)

NumPy, SciPy, MNE (EEG processing)

PyTorch (for model training and inference)

Model:

Transformer-based neural network

Trained on BCI Competition IV-2a dataset

**🚀 Installation**
Clone the Repository
git clone https://github.com/RuthBiney/BCI-Ampe_Game.git
cd BCI-Motor-Imagery

**Create a Python Virtual Environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

**Install Dependencies**
pip install -r backend/requirements.txt

**Run Backend Server**
cd backend
python app.py

**Model Details**
Architecture: Transformer with temporal attention

Dataset: BCI Competition IV-2a

Input Features: EEG time-series signals from 22 channels

Output: 4-class classification (left hand, right hand, feet, tongue)

Evaluation Metrics:

Precision: 0.8289

Recall: 0.8291

F1-score: 0.8287

Confusion Matrix included in results analysis

**✅ Testing Summary**
Validation Testing:

100% detection of invalid email format

Required fields flagged when empty

Password length and confirmation validations enforced

Integration Testing:

Uploading files triggers instant predictions

Prediction output is properly linked to file and session

Responsive error messages shown when file is invalid

Functional/System Testing:

Real-time predictions delivered within ~2 seconds

Model robust to different EEG samples

Clean UI navigation between pages

Acceptance Testing:

Users found the UI intuitive and smooth

Minor improvements suggested for responsiveness on smaller screens

**Acknowledgements**
BCI Competition IV-2a for providing the dataset

Open-source contributors to MNE, PyTorch, and EEG toolkits

Mentors and faculty who provided feedback during the project

Everyone who tested and reviewed the interface
