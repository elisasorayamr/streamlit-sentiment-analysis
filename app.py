#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import joblib
import streamlit as st
import nltk

# Download NLTK punkt tokenizer data
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import contractions
from nltk.stem import WordNetLemmatizer

# Define the preprocess_text function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove links, mentions, non-ASCII characters, punctuations
    text = re.sub(r"(?:\@|https?\://)\S+|[^a-zA-Z\s]|[\u0080-\uffff]|["+string.punctuation+"]", "", text)
    # Expand contractions
    text = contractions.fix(text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove short words
    tokens = [word for word in tokens if len(word) > 2]
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a function to preprocess and vectorize text
def preprocess_and_vectorize(text):
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    return text_vector

# Define a function to make predictions
def predict_sentiment(text):
    text_vector = preprocess_and_vectorize(text)
    prediction = svm_model.predict(text_vector)
    return prediction[0]

# Streamlit app
st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        if prediction == 1:
            st.write("Positive sentiment")
        else:
            st.write("Negative sentiment")
    else:
        st.write("Please enter some text.")
