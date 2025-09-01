import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.stem import PorterStemmer
import re
import time
import os

# Set the NLTK data path
# nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

with open('rfc_sentiment_model', 'rb') as file:
    model = pickle.load(file)



 
st.set_page_config(layout='wide',page_title='Sentiment Analysis')
st.title(':blue[Twitter Sentiment Analysis]')

# function for text preprocessing
ps = PorterStemmer()

def preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    full_txt = []
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            full_txt.append(ps.stem(i))
    return ' '.join(full_txt)

def pre(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text

def predict(text):
    prediction = model.predict([text])
    return prediction[0]  



st.write('''
         Tweet Sentiment Analysis refers to process of determining the emotional tone sentiment expressed in tweet.
         ''')

st.markdown("check out kaggle notebook [link](https://www.kaggle.com/code/gauravbosamiya/twitter-sentiment-analysis)")
user_input = st.text_area("Enter text to analyze", placeholder="Enter text to analyze....",label_visibility="hidden")
user_input = preprocessing(user_input)
user_input = pre(user_input)


# Positive - 1
# Negative - 0
# Neutral - 2
# Irrelevant - 3


if st.button(':orange[Predict]'):
    if user_input:
        start_time = time.time()
        
        with st.spinner('Analyzing...'):
            time.sleep(1)
            prediction = predict(user_input)
            
        end_time = time.time()
        time_taken = end_time - start_time
        
        if prediction == 0:
            st.subheader('Prediction :  :red[Negative]')
        elif prediction == 1:
            st.subheader('Prediction :  :green[Positive]')
            st.balloons()
        elif prediction == 2:
            st.subheader('Prediction :  :orange[Neutral]')
        else:
            st.subheader('Prediction :  :red[Irrelevant]')
            
        st.write(f"Time taken for prediction {time_taken:.2f} seconds")

    else:
        st.write('Please enter some text to analyze.')
