# Import necessary libraries and modules
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD


# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
intents =json.loads(open("intents.json").read())

# Initialize empty lists for words, classes, and documents
words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']


# Process each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in each pattern
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add tokenized words and corresponding intent tag to documents
        documents.append((word_list,intent['tag']))
        # Add intent tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Lemmatize words and remove ignored letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Sort and remove duplicates from words and classes 
words =sorted(set(words))
classes=sorted(set(classes))


# Save words and classes to pickle files
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# Initialize training data and create bag of words
training =[]
output_empty = [0]* len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create output row with 1 at the index of the class
    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])

# Shuffle and convert training data to numpy array
random.shuffle(training)
training=np.array(training, dtype="object")

# Split the training data into input and output
train_x=list(training[:,0])
train_y=list(training[:,1])


# Build the neural network model
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# Compile the model using Stochastic Gradient Descent optimizer
sgd = SGD(learning_rate=0.01,momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

# Train the model with the training data
hist = model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbotModel.h5',hist)
print("Done")



