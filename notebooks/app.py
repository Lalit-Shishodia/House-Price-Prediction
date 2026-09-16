import sys
from pathlib import Path

import streamlit as st


# Add src directory to Python path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from predict import predict_price


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below to estimate "
    "the house price using a trained machine-learning model."
)

st.divider()


# --------------------------------------------------
# Input form
# --------------------------------------------------

with st.form("house_price_form"):

    st.subheader("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area",
            min_value=1,
            value=3000,
            step=100,
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            value=3,
            step=1,
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            value=2,
            step=1,
        )

        stories = st.number_input(
            "Stories",
            min_value=0,
            value=2,
            step=1,
        )

    with col2:
        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            value=1,
            step=1,
        )

        mainroad = st.selectbox(
            "Main Road",
            ["yes", "no"],
        )

        guestroom = st.selectbox(
            "Guestroom",
            ["yes", "no"],
        )

        basement = st.selectbox(
            "Basement",
            ["yes", "no"],
        )

    with col3:
        hotwaterheating = st.selectbox(
            "Hot Water Heating",
            ["yes", "no"],
        )

        airconditioning = st.selectbox(
            "Air Conditioning",
            ["yes", "no"],
        )

        prefarea = st.selectbox(
            "Preferred Area",
            ["yes", "no"],
        )

        furnishingstatus = st.selectbox(
            "Furnishing Status",
            [
                "furnished",
                "semi-furnished",
                "unfurnished",
            ],
        )

    submitted = st.form_submit_button(
        "Predict House Price"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }

    try:
        prediction = predict_price(input_data)

        st.success("Prediction generated successfully.")

        st.metric(
            label="Estimated House Price",
            value=f"{prediction:,.2f}",
        )

        st.info(
            "This is a machine-learning estimate and "
            "should not be treated as a certified property valuation."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )