import streamlit as st
from transformers import pipeline

# Set the title for the app
st.title("Hugging Face Text Classification")

# Sidebar to select a model
model_name = st.sidebar.selectbox(
    "Select Pre-trained Model", 
    ("distilbert-base-uncased", "bert-base-uncased", "roberta-base")
)

# Load the Hugging Face model and tokenizer for text classification
@st.cache_resource
def load_model(model_name):
    classifier = pipeline("text-classification", model=model_name)
    return classifier

# Load the selected model
classifier = load_model(model_name)

# Get user input for classification
st.write(f"### Text Classification using {model_name}")
text_input = st.text_area("Enter text to classify:")

if text_input:
    # Perform classification
    result = classifier(text_input)
    
    # Show the result
    st.write("### Classification Result")
    st.write(f"Label: {result[0]['label']}")
    st.write(f"Confidence: {result[0]['score']:.4f}")

