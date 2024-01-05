# Import necessary libraries and modules
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model


# Initialize WordNet lemmatizer, load intents, words, and the trained model
lemmatizer=WordNetLemmatizer()
intents =json.loads(open("intents.json").read())

words=pickle.load(open("words.pkl",'rb'))
classes=pickle.load(open("classes.pkl",'rb'))
model=load_model('chatbotModel.h5')


# Function to clean up a sentence by tokenizing and lemmatizing words
def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Function to create a bag of words representation for a sentence
def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag = [0]*len(words)
    # Set corresponding index to 1 if the word is present in the sentence
    for w in sentence_words:
        for i , word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)


# Function to predict the intent of a given sentence
def predict_class(sentence):
    bow=bag_of_words(sentence)
    # Use the trained model to predict the probabilities of each intent
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    # Extract intents with probability greater than the error threshold
    results= [[i,r] for i , r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent': classes[r[0]],'probability': str(r[1])})
    return return_list


# Function to get a random response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    # Match the predicted intent tag with predefined intents and choose a random response
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot running")

# Main loop for running the chatbot
while True:
    message = input("")
    # Predict the intent of the user's message
    ints=predict_class(message)
    # Get a response based on the predicted intent
    res=get_response(ints,intents)
    print(res)


