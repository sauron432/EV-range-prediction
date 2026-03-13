# ⚡ EV Range Predictor

A machine learning project that predicts the driving range (in km) of electric vehicles (EVs) based on key specifications such as top speed, battery capacity, torque, acceleration, and fast charge power.

This project includes a trained Random Forest model served via a **FastAPI backend** and an interactive **Streamlit frontend**, fully containerized with **Docker**.

---

## 🚀 Project Overview

The goal of this project is to estimate the range of an electric vehicle using its specifications.  
It demonstrates the complete ML lifecycle — from data preprocessing and model training to containerized deployment using FastAPI and Streamlit.

---

## 📂 Project Structure

```
EV-range-prediction/
│
├── api/                        # FastAPI backend
├── streamlit/                  # Streamlit frontend app
├── src/                        # Source code (preprocessing, model logic)
├── model/                      # Trained model and scaler files
├── data/                       # Dataset files
├── notebook/                   # Jupyter notebooks for EDA and training
├── main.py                     # Entry point
├── Dockerfile.api              # Dockerfile for FastAPI service
├── Dockerfile.streamlit        # Dockerfile for Streamlit service
├── docker-compose.yml          # Orchestrates API + Streamlit containers
├── requirements.txt            # Shared dependencies
├── requirements.api.txt        # API-specific dependencies
├── requirements.streamlit.txt  # Streamlit-specific dependencies
├── Documentation.pdf           # Project documentation
└── README.md                   # Project documentation
```

---

## 🧠 Machine Learning Workflow

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
- Achieved R² ≈ 0.96 on test data
- Model serialized using `pickle`

---

## 🐳 Docker Setup

This project runs as two services orchestrated via Docker Compose:

| Service | Description | Port |
|---|---|---|
| `api` | FastAPI backend serving predictions | `8000` |
| `streamlit` | Streamlit frontend for user interaction | `8501` |

### Run with Docker Compose

```bash
docker-compose up --build
```

Then open:
- Streamlit app: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

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
- Predicted EV range (km)
- User input summary table

---

## 📊 Results

- **Test R² Score:** 96%
- **Train R² Score:** 89%
- Real-time predictions delivered via a simple and interactive web interface

---

## 🧰 Technologies Used

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- FastAPI
- Streamlit
- Docker & Docker Compose

---

## 👤 Author

**Bishesh Khadgi**  
📧 [bkhadgi7@gmail.com](mailto:bkhadgi7@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/bishesh-khadgi-b884462a3/)