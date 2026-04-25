import streamlit as st
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
