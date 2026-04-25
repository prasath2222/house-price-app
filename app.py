import streamlit as st
from xgboost import XGBRegressor

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("Enter house details to estimate price using a machine learning model.")

# -------------------- LOAD MODEL --------------------
model = XGBRegressor()
model.load_model("model.json")  # must be in same repo

# -------------------- INPUT UI --------------------
col1, col2 = st.columns(2)

with col1:
    bed = st.number_input("Bedrooms", min_value=0, step=1)

with col2:
    bath = st.number_input("Bathrooms", min_value=0, step=1)

acre = st.number_input("Acre lot", min_value=0.0, step=0.01)
size = st.number_input("House size (sqft)", min_value=0, step=10)

# -------------------- PREDICTION --------------------
if st.button("Predict"):
    if size == 0:
        st.error("House size must be greater than 0")
    else:
        data = [[bed, bath, acre, size]]
        price = model.predict(data)[0]

        st.success(f"Predicted Price: ${price:,.2f}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Model: XGBoost Regressor | Built with Streamlit")import streamlit as st
import xgboost as xgb

# -------------------------------
# LOAD MODEL
# -------------------------------
model = xgb.Booster()
model.load_model("model.json")

# -------------------------------
# UI
# -------------------------------
st.title("🏠 House Price Predictor")

bed = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
acre = st.number_input("Acre lot", min_value=0.01)
size = st.number_input("House size (sqft)", min_value=100)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict"):

    # CREATE DATA WITH CORRECT FEATURE NAMES
    data = xgb.DMatrix(
        [[bed, bath, acre, size]],
        feature_names=['bed', 'bath', 'acre_lot', 'house_size']
    )

    # PREDICT
    price = model.predict(data)[0]

    # SAFETY (no negative price)
    if price < 0:
        price = 0

    # OUTPUT
    st.success(f"Predicted Price: ${price:,.2f}")
