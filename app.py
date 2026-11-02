
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/machine_failure_model.pkl")
threshold = joblib.load("models/failure_threshold.pkl")

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #17365D;
}
.stButton button {
    width: 100%;
    height: 3em;
    background-color: #17365D;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚙️ Predictive Maintenance System")
st.write("AI-powered machine failure detection using sensor readings.")

st.divider()

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write("""
This application uses a Random Forest classifier
trained on the AI4I 2020 Predictive Maintenance dataset.
""")

st.sidebar.metric("Classification Threshold", threshold)

# Input section
st.subheader("Machine Sensor Readings")
st.caption("Enter the current operating parameters of your machine.")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input(
        "Air Temperature (K)", 290.0, 310.0, 300.0
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature (K)", 300.0, 320.0, 310.0
    )
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)", 1000, 3000, 1500
    )

with col3:
    torque = st.number_input(
        "Torque (Nm)", 3.0, 80.0, 40.0
    )
    tool_wear = st.number_input(
        "Tool Wear (min)", 0, 300, 100
    )

st.divider()

# Derived feature preview
temperature_difference = process_temp - air_temp

mechanical_power = (
    2 * np.pi * rotational_speed * torque
) / (60 * 1000)

st.subheader("Calculated Parameters")

m1, m2 = st.columns(2)
m1.metric("Temperature Difference", f"{temperature_difference:.2f} K")
m2.metric("Mechanical Power", f"{mechanical_power:.2f} kW")

st.divider()

# Prediction
if st.button("🔍 Predict Machine Failure"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": float(air_temp),
        "Process temperature [K]": float(process_temp),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
        "Temperature Difference": float(temperature_difference),
        "Mechanical Power [kW]": float(mechanical_power)
    }])

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("Prediction Results")

    st.metric(
        "Predicted Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Potential Machine Failure Detected")
        st.write("The predicted probability exceeds the selected threshold.")
    else:
        st.success("✅ No Machine Failure Predicted")
        st.write("The predicted probability is below the selected threshold.")

    st.caption(
        f"Threshold: {threshold:.2f} | "
        f"Model: Random Forest"
    )

    st.info(
        "This is a machine-learning prediction, not a guarantee "
        "of machine condition. Verify alerts through proper inspection."
    )

st.divider()

st.caption("Machine Failure Prediction System | AI4I 2020 Dataset")
