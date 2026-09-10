"""Streamlit interface for the supply-chain regression and classification models."""

from pathlib import Path

import pandas as pd
import streamlit as st

from models import (
    prepare_data,
    train_classification_models,
    train_regression_models,
)


DEFAULT_CSV = Path(__file__).with_name("dynamic_supply_chain_logistics_dataset.csv")

st.set_page_config(page_title="Supply Chain Models", page_icon="📦", layout="wide")
st.title("Supply Chain Model Dashboard")
st.caption("Regularized regression and balanced classification")

uploaded_file = st.file_uploader("Upload the supply-chain CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file).drop(columns=["timestamp"]).dropna()
elif DEFAULT_CSV.exists():
    data = prepare_data(str(DEFAULT_CSV))
else:
    data = None

if data is None:
    st.info("Upload dynamic_supply_chain_logistics_dataset.csv to continue.")
    st.stop()

st.write(f"Rows used for training: **{len(data):,}**")

with st.spinner("Training models..."):
    regression_models, regression_results = train_regression_models(data)
    classification_models, classification_results = train_classification_models(data)

regression_table = pd.DataFrame(regression_results).T.sort_values("R2 Score", ascending=False)
classification_table = pd.DataFrame({
    name: {
        "Train Accuracy": values["Train Accuracy"],
        "Test Accuracy": values["Test Accuracy"],
    }
    for name, values in classification_results.items()
}).T.sort_values("Test Accuracy", ascending=False)

regression_tab, classification_tab, prediction_tab = st.tabs(
    ["Regression", "Classification", "Prediction"]
)

with regression_tab:
    st.subheader("Delivery-time deviation")
    st.dataframe(regression_table, use_container_width=True)
    best_regression = regression_table.index[0]
    st.success(f"Best regression model: {best_regression}")

with classification_tab:
    st.subheader("Risk classification")
    st.dataframe(classification_table, use_container_width=True)
    best_classification = classification_table.index[0]
    st.success(f"Best classification model: {best_classification}")

with prediction_tab:
    st.subheader("Predict one record")
    task = st.radio("Prediction type", ["Regression", "Classification"], horizontal=True)

    if task == "Regression":
        feature_columns = data.drop(
            columns=["delivery_time_deviation", "risk_classification"]
        ).columns
        selected_model = st.selectbox("Regression model", list(regression_models))
        input_values = {}
        columns = st.columns(3)
        for index, column in enumerate(feature_columns):
            with columns[index % 3]:
                input_values[column] = st.number_input(
                    column,
                    value=float(data[column].median()),
                )

        if st.button("Predict delivery-time deviation"):
            input_frame = pd.DataFrame([input_values])
            prediction = regression_models[selected_model].predict(input_frame)[0]
            st.metric("Predicted deviation", f"{prediction:.3f}")

    else:
        feature_columns = data.drop(
            columns=["risk_classification", "delivery_time_deviation"]
        ).columns
        selected_model = st.selectbox("Classification model", list(classification_models))
        input_values = {}
        columns = st.columns(3)
        for index, column in enumerate(feature_columns):
            with columns[index % 3]:
                input_values[column] = st.number_input(
                    column,
                    value=float(data[column].median()),
                )

        if st.button("Predict risk classification"):
            input_frame = pd.DataFrame([input_values])
            prediction = classification_models[selected_model].predict(input_frame)[0]
            st.metric("Predicted risk", str(prediction))
