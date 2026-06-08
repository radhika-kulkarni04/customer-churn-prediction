import streamlit as st
import pickle
import pandas as pd
import os

# =======================================
# Paths
# =======================================
MODEL_PATH = "churn_model.pkl"
THRESHOLD_PATH = "chosen_threshold.pkl"

# =======================================
# Load model & threshold
# =======================================
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    threshold = 0.45  # fallback
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'rb') as f:
            threshold = pickle.load(f)
    
    return model, threshold

model, threshold = load_assets()

# =======================================
# Label mappings — MUST match your training LabelEncoders
# =======================================
# Run this in your notebook to get exact values:
# for col, le in label_encoders.items():
#     print(f"'{col}': {dict(zip(le.classes_, range(len(le.classes_))))},")
label_mappings = {
    'gender':              {'Female': 0, 'Male': 1},
    'Partner':             {'No': 0, 'Yes': 1},
    'Dependents':          {'No': 0, 'Yes': 1},
    'PhoneService':        {'No': 0, 'Yes': 1},
    'InternetService':     {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity':      {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport':         {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingTV':         {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract':            {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaymentMethod':       {'Bank transfer (automatic)': 0,
                            'Credit card (automatic)': 1,
                            'Electronic check': 2,
                            'Mailed check': 3},
    # Add StreamingMovies, PaperlessBilling, etc. if your model uses them
}

# =======================================
# App UI
# =======================================
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("Customer Churn Prediction App")
st.markdown("Fill in customer details to see churn risk.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=list(label_mappings['gender'].keys()))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", options=list(label_mappings['Partner'].keys()))
    dependents = st.selectbox("Dependents", options=list(label_mappings['Dependents'].keys()))
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", options=list(label_mappings['PhoneService'].keys()))

with col2:
    internet_service = st.selectbox("Internet Service", options=list(label_mappings['InternetService'].keys()))
    online_security = st.selectbox("Online Security", options=list(label_mappings['OnlineSecurity'].keys()))
    tech_support = st.selectbox("Tech Support", options=list(label_mappings['TechSupport'].keys()))
    streaming_tv = st.selectbox("Streaming TV", options=list(label_mappings['StreamingTV'].keys()))
    contract = st.selectbox("Contract", options=list(label_mappings['Contract'].keys()))
    payment_method = st.selectbox("Payment Method", options=list(label_mappings['PaymentMethod'].keys()))
    monthly_charges = st.slider("Monthly Charges ($)", 18.25, 118.75, 70.0, step=0.05)

# Prepare input
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
}])

# Encode using mappings
for col, mapping in label_mappings.items():
    if col in input_data.columns:
        input_data[col] = input_data[col].map(mapping)

# Predict
if st.button("Predict Churn", type="primary", use_container_width=True):
    try:
        prob = model.predict_proba(input_data)[0][1]  # prob of churn
        will_churn = prob >= threshold

        if will_churn:
            st.error(f"**High churn risk** — Probability: **{prob:.1%}**")
        else:
            st.success(f"**Low churn risk** — Probability: **{prob:.1%}**")

        st.info(f"Decision threshold used: **{threshold}** (higher probability → churn)")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")