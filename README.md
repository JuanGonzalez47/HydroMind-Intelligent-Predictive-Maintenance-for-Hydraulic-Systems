# HydroMind: Intelligent Predictive Maintenance for Hydraulic Systems

HydroMind is an advanced Industrial IoT solution designed to monitor, diagnose, and predict failures in complex hydraulic systems. By leveraging machine learning models specialized by component, HydroMind transforms raw sensor data into actionable insights, minimizing downtime and optimizing operational efficiency.

---

## 🏗️ Architecture: Medallion Pipeline

HydroMind follows a robust **Medallion Data Architecture** to ensure data quality, reproducibility, and real-time inference capabilities:

### 🥉 Bronze Layer (Raw)

Contains the original 17-sensor telemetry data acquired from hydraulic systems at high frequency (60-second cycles).

### 🥈 Silver Layer (Cleaned)

Processed data where signal cleaning, missing value treatment, outlier handling, and standardization are performed to ensure analytical consistency.

### 🥇 Gold Layer (Optimized)

Feature-engineered dataset composed of 102 statistical features per cycle. Recursive Feature Elimination (RFE) is applied to retain only the most relevant features for each hydraulic component.

---

## 📂 Repository Structure

```text
hydromind-predictive-maintenance/
├── data/
│   ├── bronze/                  # Raw telemetry data
│   ├── silver/                  # Cleaned and standardized data
│   └── gold/                    # Feature-engineered datasets
│
├── models/                      # Trained ML models and scalers
│   ├── cooler/
│   ├── valve/
│   ├── pump/
│   └── accumulator/
│
├── notebooks/                   # Data analysis and experimentation
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── app/
│   └── dash_app.py              # Real-time monitoring dashboard
│
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or higher
* Git
* Virtual environment (recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/hydromind-predictive-maintenance.git
cd hydromind-predictive-maintenance
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Overview

The project uses the Hydraulic Systems Condition Monitoring Dataset, which contains telemetry collected from industrial hydraulic systems.

### Sensors Included

* Pressure Sensors (PS1–PS6)
* Temperature Sensors (TS1–TS4)
* Volume Flow Sensors (FS1–FS2)
* Motor Power Sensor (EPS1)
* Vibration Sensor (VS1)
* Cooling Efficiency
* Valve Conditions
* Pump Leakage Indicators

### Target Components

HydroMind predicts the health condition of:

* Cooler
* Valve
* Internal Pump Leakage
* Hydraulic Accumulator
* Stable Flag

---

## 🤖 Machine Learning Pipeline

The predictive maintenance workflow consists of:

1. Data acquisition and ingestion
2. Signal cleaning and preprocessing
3. Statistical feature extraction
4. Feature selection using Recursive Feature Elimination (RFE)
5. Model training and validation
6. Real-time inference and visualization

### Implemented Models

* Random Forest
* Desicion Tree
* XGBoost
* Support Vector Machines (SVM)

Each hydraulic component has its own specialized predictive model to maximize diagnostic performance.

---

## 📈 Dashboard Features

HydroMind includes an interactive dashboard built with Plotly Dash for real-time monitoring and diagnostics.

### Available Functionalities

* Real-time sensor visualization
* Component health monitoring
* Predictive maintenance alerts
* Historical trend analysis
* Feature importance visualization
* Failure risk assessment

### Launching the Dashboard

Navigate to the application directory:

```bash
cd app
```

Run the dashboard:

```bash
python dash_app.py
```

Open your browser and access:

```text
http://127.0.0.1:8050
```

---

## 🛠️ Key Technical Highlights

### Modular Architecture

Independent predictive models for:

* Cooler
* Valve
* Pump
* Accumulator

### Edge-Ready Inference

* Inference latency below 10 ms per cycle
* Suitable for industrial edge computing applications

### Early Fault Detection

* Predictive capability up to 15 seconds before cycle completion
* Enables proactive maintenance planning

### Explainable AI

* Recursive Feature Elimination (RFE)
* Feature importance analysis
* Improved interpretability for industrial operators

---

## 📉 Performance Metrics

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Performance is reported separately for each hydraulic component classifier.

---

## 🔬 Future Work

* Deployment on industrial edge devices
* Integration with cloud-based IoT platforms
* Real-time streaming through MQTT and Kafka
* Deep learning architectures for anomaly detection
* Digital Twin integration for hydraulic systems

---

## 📚 References

1. Helwig, N., Pignanelli, E., & Schütze, A. Condition Monitoring of Hydraulic Systems.
2. Noura, H., Theilliol, D., Ponsart, J.C., & Chamseddine, A. Fault Diagnosis and Fault-Tolerant Control.
3. Surucu, O., Bayer, C., & Schütze, A. Machine Learning Methods for Predictive Maintenance.
4. UCI Machine Learning Repository – Hydraulic Systems Condition Monitoring Dataset.

---

## 👨‍💻 Authors

Juan Pablo González Blandón.


## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it for academic and research purposes.
