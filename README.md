# Credit Card Fraud Detection

This project is a machine learning app built with Streamlit to detect credit card fraud using a trained model.  
It supports batch predictions from uploaded CSV files and visualizes fraud trends.

## Features

- Predict fraudulent transactions using a trained model (`fraud.pkl`)  
- Upload CSV files for batch fraud detection  
- Interactive charts to visualize fraud patterns over time  
- Easy-to-use web app interface powered by Streamlit

## How to Run

1. Clone this repo:

   git clone https://github.com/kale2861/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

2. Create and activate a virtual environment:

   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # macOS/Linux

3. Installl dependencies:
   pip install -r requirements.txt

4. Run Streamlit app:
   streamlit run dashboard.py

   Dataset
The original dataset used for training is Credit Card Fraud Detection Dataset.

Model
The model file fraud.pkl contains the trained fraud detection model. It was trained using a Random Forest Classifier and evaluated with precision, recall, and f1-score metrics.
   
   
