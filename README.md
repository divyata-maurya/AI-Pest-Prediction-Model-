# AI Leaf Disease Detection System

This project implements an AI-powered Leaf Disease Detection System using a Convolutional Neural Network (CNN).
Users can upload a leaf image through a Streamlit web interface, and the model predicts the disease category with high accuracy.

The project includes:

A trained CNN model

A Streamlit web application (app.py)

Jupyter notebooks for training and analysis

Clean folder structure for GitHub

Easy setup steps


# Setup and Execution

1. Prerequisites

Make sure the following are installed:

Python 3.9+

pip

Virtual environment support


2. Clone the Repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME


3. Create and Activate a Virtual Environment

Windows:

python -m venv venv
.\venv\Scripts\activate

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

 4. Install Dependencies

pip install -r requirements.txt

If you used TensorFlow, ensure it is included in your requirements.txt.

 5. Run the Streamlit Application

streamlit run app.py

The app will open in your browser at:

http://localhost:8501

🧠 Model Details: CNN Architecture

The core model, implemented inside train_model.ipynb, uses a Convolutional Neural Network with:

Convolution layers

MaxPooling layers

Dropout regularization

Fully connected dense layers

Softmax output for multi-class classification


Training Summary:

Dataset: Leaf images (healthy + various disease categories)

Preprocessing: Resizing, normalization, augmentation

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Loss


The final model is saved as:

models/leaf_disease_model.h5


---

🔍 Prediction Workflow

When a user uploads an image via Streamlit:

1. Image is preprocessed (resize → normalize)


2. Passed through CNN


3. Model outputs probability scores


4. Highest probability class is selected


5. Streamlit displays:

Predicted disease

Confidence score

Optionally recommended treatment/prevention tips


🔌 API / Streamlit Interface

Although the system is not a REST API, the Streamlit UI provides:

✔ Upload Image

st.file_uploader("Upload a leaf image")

✔ View uploaded image

✔ Run prediction

✔ Display results

Simple and user-friendly.

📁 Folder Structure

AI_Leaf_Detection/
├── app.py                        # Streamlit interface
├── models/
│   └── leaf_disease_model.h5     # Saved CNN model
├── notebooks/
│   ├── training.ipynb            # Model training notebook
│   └── exploration.ipynb         # Dataset analysis notebook
├── data/
│   └── (your datasets - optional)
├── screenshots/
│   └── demo.png                  # Streamlit app screenshot
├── requirements.txt
├── README.md
└── .gitignore


---

📸 Screenshots

<div align="center">
  <img src="screenshots/demo.png" alt="App Screenshot" width="500"/>
</div>
---

🧪 Example Predictions

Input Image	Predicted Disease	Confidence

Leaf_1.jpg	Blight	97%
Leaf_2.jpg	Rust	94%
Leaf_3.jpg	Healthy	99%



---

📌 Future Improvements

Deploy on Streamlit Cloud / Render / AWS

Add Grad-CAM heatmaps

Improve dataset with more disease classes

Use Transfer Learning (MobileNet, EfficientNet)

Build REST API version (FastAPI / Django)



---

🤝 Contribution Guidelines

1. Fork the repository


2. Create a new branch


3. Commit improvements


4. Create a Pull Request



