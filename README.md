# Fraud Detection for E-commerce and Credit Cards

This project focuses on building a robust fraud detection system to identify fraudulent activities in e-commerce transactions and credit card usage. The current phase covers comprehensive data analysis and a preprocessing pipeline.

## 🚀 Key Features

- **Exploratory Data Analysis (EDA)**: Detailed univariate and bivariate analysis to understand fraud patterns and feature distributions.
- **Geolocation Integration**: Mapping IP addresses to countries to identify high-risk geographical regions.
- **Advanced Feature Engineering**: Creation of time-based features (e.g., `time_since_signup`) and behavioral velocity features (e.g., `device_usage_count`).
- **Class Imbalance Handling**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes for better model performance.
- **Modular Pipeline**: Clean and reusable Python modules for easy integration into machine learning workflows.

## 📂 Project Structure

```text
fraud-detection/
├── data/
│   ├── raw/             # Original datasets (Fraud_Data.csv, etc.)
│   └── processed/       # Final engineered and balanced dataset
├── notebooks/
│   ├── eda-fraud-data.ipynb         # Interactive data exploration
│   └── feature-engineering.ipynb   # Feature engineering & balancing pipeline
├── src/
│   ├── preprocessing.py      # Data cleaning and IP mapping
│   ├── feature_engineering.py # Custom feature creation logic
│   └── imbalance_handler.py   # SMOTE implementation
├── .gitignore
└── requirements.txt     # Python dependencies
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

The project is designed to be interactive through Jupyter Notebooks:

1.  **Run EDA**: Open `notebooks/eda-fraud-data.ipynb` to view data insights and fraud distributions.
2.  **Generate Features**: Run `notebooks/feature-engineering.ipynb`. This notebook uses the core scripts to:
    - Clean the raw data.
    - Map IP addresses to countries.
    - Create engineered features.
    - Apply SMOTE to balance the dataset.
    - Save the result to `data/processed/balanced_fraud_data.csv`.

## 📈 Next Steps
- Implement Model Development (Training and Evaluation).
- Develop model explainability using SHAP or LIME.
- Set up a deployment pipeline for real-time fraud detection.
