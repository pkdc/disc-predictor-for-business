import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings
from utils.disc_labels import decode_disc_labels
import tensorflow as tf

# Restrict TensorFlow's memory usage
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set page configuration - this changes the browser tab title
st.set_page_config(
    page_title="DISC Personality Prediction Model",
    page_icon="😎",
    layout="wide"
)

model = load_model()

MIN_WORD_COUNT = 10
FORWARD_MARKERS = ['-----Original Message-----', '-----Forwarded by', 'Forwarded by']

def make_prediction(input_text):
    cleaned_text = clean_msg_body(input_text)

    word_count = len(cleaned_text.split())
    if word_count < MIN_WORD_COUNT:
        return None, f"Text too short after cleaning ({word_count} words). Please provide at least {MIN_WORD_COUNT} words of email content."

    has_forward = any(marker in input_text for marker in FORWARD_MARKERS)

    embeddings = get_bert_embeddings([cleaned_text])
    result = model.predict(embeddings)
    disc_labels = decode_disc_labels(result)
    return disc_labels[0], "forwarded" if has_forward else None

st.title("DISC Personality Prediction Model")

st.write("This model predicts the DISC personality type based on the text input.")

input_data = st.text_area("Enter the text to predict the DISC personality type:")

if st.button('Predict'):
    prediction, warning = make_prediction(input_data)
    if prediction is None:
        st.warning(warning)
    else:
        if warning == "forwarded":
            st.warning("This appears to be a forwarded/reply email. The model was not trained on forwarded content, so results may be less accurate.")
        st.write("Predicted DISC personality type:", prediction)
else:
    st.write("Please enter the text and click the 'Predict' button to see the prediction.")