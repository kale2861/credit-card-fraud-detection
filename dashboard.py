import streamlit as st
import pandas as pd
import joblib
import altair as alt  # <-- add this import

# Load the model
model = joblib.load('fraud_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# --- Sidebar inputs for single transaction ---
st.sidebar.header("🔎 Single Transaction Input")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.01)
time = st.sidebar.number_input("Transaction Time (seconds since first transaction)", min_value=0.0, step=1.0)
pca_features = [0.0] * 28
single_input = [time] + pca_features + [amount]
single_df = pd.DataFrame([single_input], columns=['Time', *['V'+str(i) for i in range(1,29)], 'Amount'])

if st.sidebar.button("Predict Single"):
    pred = model.predict(single_df)[0]
    proba = model.predict_proba(single_df)[0][1]
    st.write("### 📊 Single Transaction Prediction")
    st.metric(label="Fraud Probability", value=f"{proba:.2%}")
    if pred == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Seems Legitimate")

st.markdown("---")

# --- Batch predictions section ---
st.write("### 📂 Batch Predictions from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with the same features as your training data (Time, V1-V28, Amount)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    expected_cols = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

    missing_cols = [col for col in expected_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        X = data[expected_cols]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:,1]

        data['Predicted_Class'] = predictions
        data['Fraud_Probability'] = probabilities

        st.write("### 📑 Predictions Table")
        st.dataframe(data.head(20))

        fraud_count = (predictions == 1).sum()
        legit_count = (predictions == 0).sum()

        st.write("### 📊 Fraud Prediction Summary")
        st.metric("Predicted Fraudulent Transactions", fraud_count)
        st.metric("Predicted Legitimate Transactions", legit_count)

        summary_df = pd.DataFrame({
            'Class': ['Legitimate', 'Fraudulent'],
            'Count': [legit_count, fraud_count]
        }).set_index('Class')
        st.bar_chart(summary_df)

        # Altair histogram for fraud probability distribution
        hist = alt.Chart(data).mark_bar().encode(
            alt.X("Fraud_Probability", bin=alt.Bin(maxbins=30), title="Fraud Probability"),
            y='count()',
            tooltip=['count()']
        ).properties(
            width=600,
            height=300,
            title="Fraud Probability Distribution"
        )

        st.altair_chart(hist, use_container_width=True)
