import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pickle


with open('le_2.pkl','rb') as file:
    label = pickle.load(file)

with open('token_2.pkl','rb') as file:
    token_word = pickle.load(file)


model = load_model('model_review_2.h5')


st.title('Movie Review Prediction')
st.write('This model is predict your review it is positive and negative based on the context of the text')


st.write('Enter your throught about on any Movie')
user_input = st.text_input('Enter Text')

# using here tokenizer
input_token = token_word.texts_to_sequences([user_input])

# now using the pad_sequence
sequence = pad_sequences(input_token,maxlen=200)

# now put into the model
y_pred = model.predict(sequence)

if st.button('Predict'):
    if y_pred > 0.5:
        st.header('Review is: Positive')

    else:
        st.header('Review is: Negative')

