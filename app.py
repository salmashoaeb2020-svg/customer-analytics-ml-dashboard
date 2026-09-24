import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="Customer & Sales Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# عنوان التطبيق
st.title("🛒 Smart Customer & Sales Analytics Dashboard")
st.markdown("An Interactive Machine Learning Application for Customer Segmentation and Sales Prediction.")

# تحميل النماذج والداتا
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    cluster_model = joblib.load('models/cluster_model.pkl')
    reg_model = joblib.load('models/regression_model.pkl')
    clf_model = joblib.load('models/classification_model.pkl')
    return scaler, cluster_model, reg_model, clf_model

@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    df.columns = ['CustomerID', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']
    return df

try:
    scaler, cluster_model, reg_model, clf_model = load_models()
    df = load_data()
    
    # القائمة الجانبية للتنقل بين أجزاء المشروع
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to", ["Overview & Data", "Customer Clustering", "Sales/Spending Regression", "Customer Classification"])

    # ------------------ الصفحة الأولى: نظرة عامة ------------------
    if page == "Overview & Data":
        st.header("📊 Dataset Overview")
        st.write("Preview of the Customer Dataset:")
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Average Age", f"{df['Age'].mean():.1f} yrs")
        col3.metric("Avg Annual Income", f"${df['Annual_Income'].mean():.1f}k")

    # ------------------ الصفحة الثانية: Clustering ------------------
    elif page == "Customer Clustering":
        st.header("🎯 Customer Segmentation (Clustering)")
        st.write("Group customers into distinct clusters based on Age, Income, and Spending Score.")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income ($k)", min_value=10, max_value=200, value=50)
        spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        
        if st.button("Predict Cluster"):
            input_data = scaler.transform([[age, income, spending]])
            cluster_id = cluster_model.predict(input_data)[0]
            st.success(f"This customer belongs to **Cluster #{cluster_id}**")

    # ------------------ الصفحة الثالثة: Regression ------------------
    elif page == "Sales/Spending Regression":
        st.header("📈 Predict Spending Score (Regression)")
        st.write("Predict spending behavior based on Age and Income.")
        
        age = st.slider("Age", 18, 80, 25)
        income = st.slider("Annual Income ($k)", 10, 150, 60)
        
        if st.button("Predict Spending Score"):
            pred_score = reg_model.predict([[age, income]])[0]
            st.info(f"Predicted Spending Score: **{pred_score:.2f} / 100**")

    # ------------------ الصفحة الرابعة: Classification ------------------
    elif page == "Customer Classification":
        st.header("🏷️ Customer Spender Category (Classification)")
        st.write("Classify if a customer is likely to be a High Spender (> 50) or Low Spender.")
        
        age = st.number_input("Age ", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($k) ", min_value=10, max_value=200, value=75)
        
        if st.button("Classify Customer"):
            pred_class = clf_model.predict([[age, income]])[0]
            if pred_class == 1:
                st.success("Result: **High Spender** 🌟")
            else:
                st.warning("Result: **Low Spender** 📉")

except Exception as e:
    st.error(f"Error loading files or models: {e}")
    st.info("Make sure all .pkl files exist in the 'models/' folder and 'Mall_Customers.csv' is in the main directory.")
