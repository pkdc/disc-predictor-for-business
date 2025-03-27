import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page configuration - this changes the browser tab title
st.set_page_config(
    page_title="DISC Personality Prediction Model",
    page_icon="🧊",
    layout="wide"
)

# Load the neural network model
## By loading the model once and reusing it, you avoid having multiple copies of the same model in memory.
@st.cache_resource
def load_model():
    # Clear session again right before loading
    return joblib.load("models/xgb_model.joblib")

model = load_model()