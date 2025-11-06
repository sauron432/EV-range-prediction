⚡ EV Range Predictor

A machine learning project that predicts the driving range (in km) of electric vehicles (EVs) based on key specifications such as top speed, battery capacity, torque, acceleration, and fast charge power.
This project includes a trained Random Forest model integrated into an interactive Streamlit web application.

---

🚀 Project Overview
The goal of this project is to estimate the range of an electric vehicle using its specifications.  
It demonstrates the complete ML lifecycle — from data preprocessing and model training to deployment using Streamlit.

---

 📂 Project Structure
```
EV-Range-Predictor/
│
├── app.py                     # Streamlit web application
├── EV range prediction.ipynb  # Data analysis, model training, and evaluation
├── .streamlit/
│   └── secrets.toml           # Stores secure model and scaler paths
├── RF_regressor.pkl        	 # Trained Random Forest model
├── scaler.pkl              	 # Fitted StandardScaler object
├── README.md                  # Project documentation
└── requirements.txt           # Requirements
```

---

🧠 Machine Learning Workflow

### 1. Dataset Description
- Data sourced from an Electric Vehicle specifications dataset
- Contains features like:
  - Top speed (km/h)
  - Battery capacity (kWh)
  - Torque (Nm)
  - Acceleration (0–100 s)
  - Fast charge power (kW DC)
- Target variable: EV driving range (km)

### 2. Data Preprocessing
- Missing value handling and cleaning
- Data transformation using StandardScaler
- Feature encoding where necessary
- Train-test split for model validation

### 3. Exploratory Data Analysis (EDA)
- Distribution analysis and correlation heatmaps
- Identification of key predictive variables
- Observed that battery capacity and fast charge power have the highest influence on range

### 4. Model Development
- Trained multiple models; Random Forest Regressor performed best
- Achieved high accuracy with R² ≈ 0.92
- Model serialized using `pickle`

### 5. Model Deployment
- The trained model and scaler are stored in secrets.toml and accessed securely using:
  ```
  st.secrets["MODEL_PATH"]
  st.secrets["SCALER_PATH"]
  ```
- Integrated into a Streamlit app for real-time EV range prediction

---

## 💻 Streamlit App Overview

### 🔹 Inputs
Users can adjust the following parameters via sidebar sliders:
- Top Speed (km/h)  
- Battery Capacity (kWh)  
- Torque (Nm)  
- Acceleration (0–100 s)  
- Fast Charge Power (kW DC)

### 🔹 Output
The app displays:
- Predicted EV range (km)  
- User input summary table

---

## 📊 Results
- Model achieved 96% accuracy (R² score) on test data and 89% accuracy (R² score) on training data.
- Real-time predictions delivered via a simple and interactive web interface using Streamlit.

---

## 🧰 Technologies Used
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Streamlit

---

## 👤 Author
Bishesh Khadgi
📧 Email: bkhadgi7@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/bishesh-khadgi-b884462a3/




