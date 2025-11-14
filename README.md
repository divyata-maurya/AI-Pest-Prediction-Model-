# AI Leaf Disease Detection System
## Problem Statement: 
This project implements an AI-powered Leaf Disease Detection System using a Convolutional Neural Network (CNN).
Users can upload a leaf image through a Streamlit web interface, and the model predicts the disease category with high accuracy.

The project includes:
A trained CNN model
A Streamlit web application (main.py)
Jupyter notebooks for training and analysis
Clean folder structure for GitHub
Easy setup steps


# Setup and Execution

- Prerequisites
Python 3.12.5
pip
Virtual environment support
 
-Create and Activate a Virtual Environment

Windows:
python -m venv venv
.\venv\Scripts\activate

- Install Dependencies

pip install -r requirements.txt

 - Run the Streamlit Application
streamlit run app.py

# Solution Overview
The core model, implemented inside train_model.ipynb, uses a Convolutional Neural Network with:

Convolution layers
MaxPooling layers
Dropout regularization
Fully connected dense layers
Softmax output for multi-class classification

## Training Summary:

Dataset: Leaf images (healthy + various disease categories)
Preprocessing: Resizing, normalization, augmentation
Loss Function: Categorical Crossentropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Loss

The final model is saved as:
models/leaf_disease_model.h5


## Tech Stack
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

API / Streamlit Interface

Although the system is not a REST API, the Streamlit UI provides:

✔ Upload Image
st.file_uploader("Upload a leaf image")
✔ View uploaded image
✔ Run prediction
✔ Display results

Simple and user-friendly.

# Setup Steps
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

🧪 Example Predictions

Input Image	Predicted Disease	Confidence

Leaf_1.jpg	Blight	97%
Leaf_2.jpg	Rust	94%
Leaf_3.jpg	Healthy	99%
