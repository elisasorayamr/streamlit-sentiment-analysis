#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved SVM model
with open('svm_sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Streamlit app title
st.title("Twitter Sentiment Analysis")

# Input text from user
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    # Preprocess and transform the input text
    transformed_input = tfidf_vectorizer.transform([user_input])
    
    # Make prediction
    prediction = model.predict(transformed_input)[0]
    
    # Display the result
    if prediction == 1:
        st.write("Positive sentiment")
    else:
        st.write("Negative sentiment")

