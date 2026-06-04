# Financial Fraud Detection System

A complete end-to-end machine learning solution for detecting fraudulent financial transactions. This project includes a data processing and training pipeline, a trained Random Forest model, and an interactive Streamlit dashboard for data exploration and real-time prediction.

## 🚀 Features

*   **Interactive Dashboard:** A multi-page Streamlit app for visualizing data and testing the model.
*   **Exploratory Data Analysis (EDA):** Insights into fraud rates, transaction distributions, and merchant category analysis.
*   **Real-time Prediction:** Input transaction details (Amount, Category, Location, etc.) and receive a fraud probability score instantly.
*   **Model Insights:** Visualization of feature importance, showing which factors most influence the fraud detection engine.
*   **Robust ML Pipeline:** Uses a `RandomForestClassifier` with integrated preprocessing (scaling and encoding).

## 📂 Project Structure

*   `app.py`: The main Streamlit dashboard application.
*   `train_model.py`: Script to preprocess data, train the Random Forest model, and export artifacts.
*   `requirements.txt`: Python dependencies required to run the project.
*   `model_pipeline.joblib`: The serialized machine learning pipeline (preprocessor + model).
*   `feature_info.joblib`: Metadata containing categorical options and feature names for the UI.
*   `financial_fraud_detection_dataset.csv`: The primary dataset used for training and analysis.

## 🛠️ Setup & Installation

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
The project comes with a pre-trained model. If you wish to retrain it using the dataset:
```bash
python train_model.py
```

### 3. Launch the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

## 📊 Model Performance
The current model achieves an accuracy of approximately **91%** on the test dataset, providing a strong baseline for identifying suspicious transactions while minimizing false positives.

## 📝 Technologies Used
*   **Python**
*   **Pandas & NumPy** (Data Manipulation)
*   **Scikit-Learn** (Machine Learning & Pipelines)
*   **Streamlit** (Interactive UI)
*   **Matplotlib & Seaborn** (Data Visualization)
*   **Joblib** (Model Serialization)
