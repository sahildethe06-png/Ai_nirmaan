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


def _prediction_frame(input_values: dict[str, object], model: object) -> pd.DataFrame:
    """Apply the same one-hot feature shape used while fitting a model."""
    frame = pd.get_dummies(pd.DataFrame([input_values]), dtype=float)
    feature_names = getattr(model[-1], "feature_names_in_", frame.columns)
    return frame.reindex(columns=feature_names, fill_value=0)

st.set_page_config(page_title="Supply Chain Models", page_icon="📦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #e9f0ec; --muted: #91a39a; --panel: #17231f; --line: #2a3b34; --lime: #b7e36b; }
    .stApp { background: #0b1210; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(11, 18, 16, .9); }
    [data-testid="stSidebar"] { background: #101a17; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    h1, h2, h3, p, label, [data-testid="stMetricLabel"] { font-family: 'Manrope', sans-serif; }
    h1 { letter-spacing: -1px; font-weight: 800; }
    h2, h3 { color: var(--ink); }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 8px; }
    [data-testid="stMetricValue"] { color: var(--lime); font-family: 'DM Mono', monospace; }
    .eyebrow { color: var(--lime); font: 500 12px 'DM Mono', monospace; letter-spacing: 1.5px; text-transform: uppercase; }
    .hero { border-bottom: 1px solid var(--line); padding: .5rem 0 1.6rem; margin-bottom: 1.3rem; }
    .hero p { color: var(--muted); margin: .4rem 0 0; }
    .status { color: var(--lime); font: 500 12px 'DM Mono', monospace; }
    .section-note { color: var(--muted); font-size: .88rem; }
    .stButton > button { background: var(--lime); color: #101810; border: 0; font-weight: 700; border-radius: 5px; }
    .stButton > button:hover { background: #d2f493; color: #101810; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='eyebrow'>NORTHSTAR / ML OPS</div>", unsafe_allow_html=True)
    st.markdown("## Supply chain lab")
    st.caption("Model intelligence for delivery risk and fulfillment performance.")
    st.divider()
    st.markdown("### Data source")
    uploaded_file = st.file_uploader("Upload a training CSV", type="csv", label_visibility="collapsed")
    st.markdown("<span class='section-note'>Required: delivery_time_deviation and risk_classification</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<span class='status'>● SYSTEM READY</span>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">SUPPLY CHAIN / DECISION CONSOLE</div>
      <h1>Know what moves late.</h1>
      <p>Train, compare, and interrogate delivery models from one operational view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file).drop(columns=["timestamp"], errors="ignore").dropna()
elif DEFAULT_CSV.exists():
    data = prepare_data(str(DEFAULT_CSV))
else:
    data = None

if data is None:
    st.info("Upload a supply-chain CSV from the control rail to begin.")
    st.stop()

try:
    with st.spinner("Training models..."):
        regression_models, regression_results = train_regression_models(data)
        classification_models, classification_results = train_classification_models(data)
except ValueError as error:
    st.error(str(error))
    st.stop()

best_regression = max(regression_results, key=lambda name: regression_results[name]["R2 Score"])
best_classification = max(classification_results, key=lambda name: classification_results[name]["Test Accuracy"])
feature_count = len(data.drop(columns=["delivery_time_deviation", "risk_classification"]))

metric_columns = st.columns(4)
metric_columns[0].metric("Records analyzed", f"{len(data):,}")
metric_columns[1].metric("Input signals", feature_count)
metric_columns[2].metric("Best R²", f"{regression_results[best_regression]['R2 Score']:.2f}")
metric_columns[3].metric("Best risk accuracy", f"{classification_results[best_classification]['Test Accuracy']:.1%}")

regression_table = pd.DataFrame(regression_results).T.sort_values("R2 Score", ascending=False)
classification_table = pd.DataFrame({
    name: {
        "Train Accuracy": values["Train Accuracy"],
        "Test Accuracy": values["Test Accuracy"],
    }
    for name, values in classification_results.items()
}).T.sort_values("Test Accuracy", ascending=False)

regression_tab, classification_tab, prediction_tab = st.tabs(
    ["01 / Deviation model", "02 / Risk model", "03 / Single prediction"]
)

with regression_tab:
    st.subheader("Delivery-time deviation")
    st.markdown("<p class='section-note'>How accurately can each model estimate the difference between planned and actual delivery time?</p>", unsafe_allow_html=True)
    st.dataframe(regression_table, width="stretch")
    st.success(f"Leading model  /  {best_regression}")

with classification_tab:
    st.subheader("Risk classification")
    st.markdown("<p class='section-note'>Compare how reliably each classifier separates high-risk and low-risk shipments.</p>", unsafe_allow_html=True)
    st.dataframe(classification_table, width="stretch")
    st.success(f"Leading model  /  {best_classification}")

with prediction_tab:
    st.subheader("Predict one record")
    st.markdown("<p class='section-note'>Use the median profile as a baseline, then adjust the signals for one shipment.</p>", unsafe_allow_html=True)
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
                if pd.api.types.is_numeric_dtype(data[column]):
                    input_values[column] = st.number_input(column, value=float(data[column].median()))
                else:
                    input_values[column] = st.selectbox(column, sorted(data[column].astype(str).unique()))

        if st.button("Predict delivery-time deviation"):
            model = regression_models[selected_model]
            prediction = model.predict(_prediction_frame(input_values, model))[0]
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
                if pd.api.types.is_numeric_dtype(data[column]):
                    input_values[column] = st.number_input(column, value=float(data[column].median()))
                else:
                    input_values[column] = st.selectbox(column, sorted(data[column].astype(str).unique()))

        if st.button("Predict risk classification"):
            model = classification_models[selected_model]
            prediction = model.predict(_prediction_frame(input_values, model))[0]
            st.metric("Predicted risk", str(prediction))
