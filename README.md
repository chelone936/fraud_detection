# Fraud Detection for E-commerce and Credit Cards

This project implements a comprehensive fraud detection system designed to identify fraudulent activities in e-commerce and banking transactions. The solution spans from data preprocessing and geolocation mapping to ensemble modeling and SHAP-based explainability.

## 🚀 Key Features

- **Geolocation Integration**: IP addresses were mapped to countries to identify high-risk regions.
- **Advanced Feature Engineering**: Temporal features (e.g., `time_since_signup`) and behavioral velocity metrics (e.g., `device_usage_count`, `ip_usage_count`) were engineered to capture bot-like behavior.
- **Class Imbalance Management**: SMOTE was utilized to balance training data, ensuring robust detection of minority fraud cases.
- **High-Performance Modeling**: A Random Forest ensemble was developed, achieving an **F1-Score of 0.90** and an **AUC-PR of 0.97**.
- **Model Explainability (XAI)**: SHAP analysis was applied to interpret model decisions and identify primary fraud drivers.
- **Business Insights**: Actionable recommendations were formulated to mitigate risk through dynamic velocity thresholds and user sandboxing.

## 📂 Project Structure

```text
fraud-detection/
├── data/
│   ├── raw/                 # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/           # Cleaned, engineered, and balanced datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb         # Exploratory Data Analysis
│   ├── feature-engineering.ipynb    # Feature engineering & balancing
│   ├── model-training.ipynb         # Model building and evaluation
│   └── model-explainability.ipynb   # SHAP analysis & interpretation
├── src/
│   ├── preprocessing.py      # IP mapping and cleaning
│   ├── feature_engineering.py # Velocity and time feature logic
│   ├── imbalance_handler.py   # SMOTE implementation
│   └── modeling.py            # Model training and evaluation utilities
├── scripts/
│   └── update_notebook.py     # Helper for programmatically updating notebooks
├── FINAL_REPORT.md            # Comprehensive project summary and insights
├── requirements.txt           # Python dependencies
└── README.md
```

## 🛠️ Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   cd fraud-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 How to Use

The project workflow is organized into sequential Jupyter Notebooks:

1.  **Data Analysis**: Run `notebooks/eda-fraud-data.ipynb` for initial insights.
2.  **Feature Pipeline**: Execute `notebooks/feature-engineering.ipynb` to clean data and generate the balanced training set.
3.  **Model Development**: Use `notebooks/model-training.ipynb` to train the Random Forest model and view performance metrics.
4.  **Explainability**: Open `notebooks/model-explainability.ipynb` to view SHAP interpretations and business recommendations.

## 📈 Performance Summary

The final **Random Forest** model outperformed the baseline Logistic Regression across all key metrics:
- **Precision (Fraud)**: 0.96
- **Recall (Fraud)**: 0.85
- **F1-Score**: 0.90
- **AUC-PR**: 0.97
