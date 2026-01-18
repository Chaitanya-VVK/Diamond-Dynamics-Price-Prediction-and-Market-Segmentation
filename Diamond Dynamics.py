#Import Libraries & Load Models
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models & scalers
with open("rf_price_model.pkl", "rb") as f:
    price_model = pickle.load(f)

with open("scaler_reg.pkl", "rb") as f:
    scaler_reg = pickle.load(f)

with open("kmeans_cluster_model.pkl", "rb") as f:
    cluster_model = pickle.load(f)

with open("scaler_cluster.pkl", "rb") as f:
    scaler_cluster = pickle.load(f)

#App Title & Layout
st.set_page_config(page_title="Diamond Price & Market Segmentation", layout="centered")

st.title("💎 Diamond Price Prediction & Market Segmentation")
st.write("Predict diamond price and identify its market segment using ML models.")

#User Inputs
st.header("🔢 Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, step=0.01)
depth = st.number_input("Depth", min_value=40.0, max_value=80.0, step=0.1)
table = st.number_input("Table", min_value=40.0, max_value=95.0, step=0.1)

x = st.number_input("Length (x)", min_value=3.0, step=0.01)
y = st.number_input("Width (y)", min_value=3.0, step=0.01)
z = st.number_input("Depth (z)", min_value=1.0, step=0.01)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

#Encoding Maps
cut_map = {'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
color_map = {'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
clarity_map = {'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}

#Feature Engineering
if st.button("🔮 Predict"):
    carat_log = np.log1p(carat)
    volume = x * y * z
    dimension_ratio = (x + y) / (2 * z)

    cut_enc = cut_map[cut]
    color_enc = color_map[color]
    clarity_enc = clarity_map[clarity]


#Price Prediction
    X_reg = pd.DataFrame([[
        carat_log, depth, table, dimension_ratio,
        cut_enc, color_enc, clarity_enc
    ]], columns=[
        'carat_log','depth','table','dimension_ratio',
        'cut_enc','color_enc','clarity_enc'
    ])

    X_reg_scaled = scaler_reg.transform(X_reg)
    price_log_pred = price_model.predict(X_reg_scaled)[0]
    price_pred = np.expm1(price_log_pred)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {price_pred:,.2f}")

#Cluster Prediction
    X_cluster = pd.DataFrame([[
        carat_log, volume, dimension_ratio,
        cut_enc, color_enc, clarity_enc
    ]], columns=[
        'carat_log','volume','dimension_ratio',
        'cut_enc','color_enc','clarity_enc'
    ])

    X_cluster_scaled = scaler_cluster.transform(X_cluster)
    cluster_id = cluster_model.predict(X_cluster_scaled)[0]

    cluster_names = {
        0: "Premium Heavy Diamonds",
        1: "Affordable Small High-Quality Diamonds",
        2: "Mid-range Balanced Diamonds"
    }

    st.subheader("📊 Market Segment")
    st.info(cluster_names[cluster_id])
