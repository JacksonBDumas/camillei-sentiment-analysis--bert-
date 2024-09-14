import streamlit as st

# Check for required packages
try:
    from transformers import pipeline
except ImportError as e:
    st.error(f"Error importing packages: {e}")
    st.stop()

# Load the pre-trained sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

# Streamlit for real-time interaction
st.title('CamillEI: BERT Sentiment Analysis Demo')
st.write('Enter a sentence to analyze its sentiment.')

# Input text for sentiment analysis
input_text = st.text_input('Input Text', '')

if input_text:
    # Predict sentiment
    result = classifier(input_text)[0]
    sentiment = result['label']
    score = result['score']
    
    st.write(f"Sentiment: {sentiment} ({score:.2f})")
