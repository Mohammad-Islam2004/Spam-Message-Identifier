import streamlit as st
import pickle
import string
import nltk
import os

# Create a local nltk_data directory
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DATA_DIR)

nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def convert(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

vector = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    transformed_text = convert(input_sms)

    vector_input = vector.transform([transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")