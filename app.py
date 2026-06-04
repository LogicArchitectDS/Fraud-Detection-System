import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(page_title="Financial Fraud Detection", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('financial_fraud_detection_dataset.csv')
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], dayfirst=True)
    return df

@st.cache_resource
def load_model_artifacts():
    model = joblib.load('model_pipeline.joblib')
    feature_info = joblib.load('feature_info.joblib')
    return model, feature_info

df = load_data()
model, feature_info = load_model_artifacts()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard / EDA", "Fraud Prediction", "Model Insights"])

if page == "Dashboard / EDA":
    st.title("📊 Financial Fraud Detection Dashboard")
    st.write("Exploratory Data Analysis of the transaction dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraudulent Transactions", df['Fraudulent'].sum())
    col3.metric("Fraud Rate", f"{(df['Fraudulent'].mean() * 100):.2f}%")

    st.subheader("Data Overview")
    st.dataframe(df.head(10))

    st.subheader("Visualizations")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.write("**Fraud Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(x='Fraudulent', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col_v2:
        st.write("**Fraud by Merchant Category**")
        fig, ax = plt.subplots()
        sns.barplot(x='Merchant_Category', y='Fraudulent', data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.write("**Transaction Amount Distribution by Fraud**")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=df, x='Transaction_Amount', hue='Fraudulent', fill=True, ax=ax)
    st.pyplot(fig)

elif page == "Fraud Prediction":
    st.title("🔍 Real-time Fraud Prediction")
    st.write("Enter transaction details to predict if it is fraudulent.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            category = st.selectbox("Merchant Category", feature_info['cat_options']['Merchant_Category'])
            payment = st.selectbox("Payment Method", feature_info['cat_options']['Payment_Method'])
            device = st.selectbox("Device Type", feature_info['cat_options']['Device_Type'])
            location = st.selectbox("Location", feature_info['cat_options']['Location'])
            
        with col2:
            is_intl = st.selectbox("Is International?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            prev_tx = st.number_input("Previous Transactions", min_value=0, value=10)
            avg_spend = st.number_input("Average Spend", min_value=0.0, value=250.0)
            acc_age = st.number_input("Account Age (Days)", min_value=0, value=365)
            suspicious = st.selectbox("Suspicious Keyword?", feature_info['cat_options']['Suspicious_Keyword'])
            tx_date = st.date_input("Transaction Date", datetime.now())
            tx_time = st.time_input("Transaction Time", datetime.now().time())

        submit = st.form_submit_button("Predict Fraud")

    if submit:
        # Prepare input data
        dt = datetime.combine(tx_date, tx_time)
        input_df = pd.DataFrame([{
            'Transaction_Amount': amount,
            'Merchant_Category': category,
            'Payment_Method': payment,
            'Device_Type': device,
            'Location': location,
            'Is_International': is_intl,
            'Previous_Transactions': prev_tx,
            'Average_Spend': avg_spend,
            'Account_Age_Days': acc_age,
            'Suspicious_Keyword': suspicious,
            'Hour': dt.hour,
            'DayOfWeek': dt.weekday()
        }])

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"🚨 ALERT: Potential Fraud Detected! (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Transaction appears Legitimate. (Fraud Probability: {probability:.2%})")

elif page == "Model Insights":
    st.title("💡 Model Insights")
    st.write("Understanding the factors that influence the fraud detection model.")

    # Extract feature importance from the pipeline
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(feature_info['cat_cols'])
    feature_names = np.concatenate([feature_info['num_cols'], cat_feature_names])
    
    importances = classifier.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)

    st.subheader("Top 15 Most Important Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='magma', ax=ax)
    st.pyplot(fig)

    st.info("Higher importance indicates that the model relies more on that feature to distinguish between fraudulent and legitimate transactions.")
