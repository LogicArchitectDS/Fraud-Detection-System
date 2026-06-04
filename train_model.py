import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Drop IDs and handle Date
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], dayfirst=True)
    df['Hour'] = df['Transaction_Date'].dt.hour
    df['DayOfWeek'] = df['Transaction_Date'].dt.dayofweek
    
    # Define features and target
    target = 'Fraudulent'
    drop_cols = ['Transaction_ID', 'Customer_ID', 'Transaction_Date', target]
    X = df.drop(columns=drop_cols)
    y = df[target]
    
    # Identify categorical and numerical columns
    cat_cols = ['Merchant_Category', 'Payment_Method', 'Device_Type', 'Location', 'Suspicious_Keyword']
    num_cols = ['Transaction_Amount', 'Is_International', 'Previous_Transactions', 'Average_Spend', 'Account_Age_Days', 'Hour', 'DayOfWeek']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    joblib.dump(model, 'model_pipeline.joblib')
    print("\nModel saved as 'model_pipeline.joblib'")
    
    # Save feature names for UI
    feature_info = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'cat_options': {col: df[col].unique().tolist() for col in cat_cols}
    }
    joblib.dump(feature_info, 'feature_info.joblib')
    print("Feature info saved as 'feature_info.joblib'")

if __name__ == "__main__":
    train_model('financial_fraud_detection_dataset.csv')
