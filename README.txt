# training.py Code Documentation

## Libraries and Modules
- `random`: Provides functions for generating random numbers.
- `json`: Handles JSON file loading and parsing.
- `pickle`: Serializes and deserializes Python objects for storage.
- `numpy`: Library for numerical operations, used for array manipulations.
- `nltk`: Natural Language Toolkit, used for text processing tasks.
- `WordNetLemmatizer`: Part of NLTK, performs lemmatization on words.
- `keras`: Deep learning library for building neural networks.
- `Sequential`, `Dense`, `Activation`, `Dropout`: Components of a neural network.
- `SGD`: Stochastic Gradient Descent optimizer.

## Data Preprocessing
1. Load intents from "intents.json".
2. Tokenize and lemmatize words from patterns in intents.
3. Create a bag of words representation for each pattern.
4. Preprocess and organize training data (X and Y).

## Neural Network Model
- Create a sequential model with dense layers and dropout for training.
- Compile the model using categorical cross-entropy loss and SGD optimizer.
- Train the model using the preprocessed training data.

## Save Model and Preprocessed Data
- Save the trained model and preprocessed data (words and classes) to pickle files.

# Note: The code assumes the availability of the "intents.json" file for training.


# Chatbot.py Code Documentation

## Libraries and Modules
- `random`: Provides functions for generating random numbers.
- `json`: Handles JSON file loading and parsing.
- `pickle`: Serializes and deserializes Python objects for storage.
- `numpy`: Library for numerical operations, used for array manipulations.
- `nltk`: Natural Language Toolkit, used for text processing tasks.
- `WordNetLemmatizer`: Part of NLTK, performs lemmatization on words.
- `keras`: Deep learning library for building neural networks.
- `load_model`: Function to load a pre-trained neural network model.

## Initialization
- Initialize the WordNet lemmatizer, load intents from "intents.json", and load preprocessed data (words, classes, and the trained model).

## Data Processing Functions
1. `clean_up_sentence(sentence)`: Tokenizes and lemmatizes words in a given sentence.
2. `bag_of_words(sentence)`: Creates a bag of words representation for a given sentence.

## Intent Prediction Function
- `predict_class(sentence)`: Utilizes the trained neural network model to predict the intent of a given sentence.
  - Returns a list of intents along with their probabilities.

## Response Retrieval Function
- `get_response(intents_list, intents_json)`: Matches the predicted intent with predefined intents and retrieves a random response.
  - Selects a response from the intents JSON based on the predicted intent.

## Main Loop
- Continuously prompts the user for input and responds with the chatbot's prediction.
- The loop breaks when the user exits the program.

## User Interaction
- The chatbot prompts the user for input, predicts the intent, and outputs a response.

# Note: The code assumes the availability of the pre-trained model ('chatbotModel.h5') and the preprocessed data files ('words.pkl' and 'classes.pkl').


