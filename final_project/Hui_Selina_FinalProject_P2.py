# Selina Hui
# ITP 259 Fall 2024
# Final Project
# Problem 2

# RNN English to Spanish ModelTime
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras as keras
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding, LayerNormalization, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

keras.mixed_precision.set_global_policy('mixed_float16')

# Load English to Spanish Data into DataFrame
Eng_To_Span_Data = pd.read_csv("/Users/Selina/Documents/ITP259/FinalProject/english-spanish-dataset.csv")
pd.set_option("display.max_columns", None)
Eng_Span_DF = pd.DataFrame(Eng_To_Span_Data)

# Question 1: Process the dataset into English sentences and Spanish sentences, reducing the size to first 50,000 sentences.
# Split DataFrame into English and Spanish sentences
X_English = Eng_Span_DF.iloc[:50000,1]
y_Spanish = Eng_Span_DF.iloc[:50000,2]

# Clean and Preprocess English and Spanish sentences
X_English = [str(text).lower().strip() for text in X_English]
y_Spanish = [str(text).lower().strip() for text in y_Spanish]

# Convert to numpy arrays for model input
X_English = np.array(X_English)
y_Spanish = np.array(y_Spanish)

# Print Sample Translations of English to Spanish
for sample_i in range(3):
    print('English sample {}:  {}'.format(sample_i + 1, X_English[sample_i]))
    print('Spanish sample {}:  {}\n'.format(sample_i + 1, y_Spanish[sample_i]))

# Count vocab frequency for English and Spanish
english_words_counter = collections.Counter([word for sentence in X_English for word in sentence.split()])
spanish_words_counter = collections.Counter([word for sentence in y_Spanish for word in sentence.split()])

# Print English and Spanish Language Observations
print('\n{} English words.'.format(len([word for sentence in X_English for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('\n10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')

print('\n{} Spanish words.'.format(len([word for sentence in y_Spanish for word in sentence.split()])))
print('{} unique Spanish words.'.format(len(spanish_words_counter)))
print('\n10 Most common words in the Spanish dataset:')
print('\n"' + '" "'.join(list(zip(*spanish_words_counter.most_common(10)))[0]) + '"')

# Question 2: Tokenize Sentences in English and Spanish
# Define Tokenizer Function
def create_tokenizer(text, num_words=None):
    text_tokenizer = Tokenizer(
        num_words=num_words,
        lower=True,
    )
    text_tokenizer.fit_on_texts(text)
    return text_tokenizer.texts_to_sequences(text), text_tokenizer

# Tokenize English and Spanish sentences
english_tokenizer = Tokenizer(oov_token='<OOV>')
spanish_tokenizer = Tokenizer(oov_token='<OOV>')

english_tokenizer.fit_on_texts(X_English)
spanish_tokenizer.fit_on_texts(y_Spanish)

english_sequences = english_tokenizer.texts_to_sequences(X_English)
spanish_sequences = spanish_tokenizer.texts_to_sequences(y_Spanish)

# Question 3: Pad sentences
# Define function to pad sentences to a given length
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

# Define Preprocessing Function
def preprocess(sentence_x, sentence_y):
    preprocess_x, x_tk = create_tokenizer(sentence_x)
    preprocess_y, y_tk = create_tokenizer(sentence_y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    return preprocess_x, preprocess_y, x_tk, y_tk

# Use preprocessing function on English and Spanish sentences
preproc_english_sentences, preproc_spanish_sentences, english_tokenizer, \
spanish_tokenizer = preprocess(X_English, y_Spanish)

# Count English and Spanish Sequence Length and Vocab Size
max_english_length = preproc_english_sentences.shape[1]
max_spanish_length = preproc_spanish_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
spanish_vocab_size = len(spanish_tokenizer.word_index)

# Hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 32
epochs = 10

# Question 4: Define and Build Simple RNN Translation Model
def simple_translation_model(input_shape, output_length, english_vocab_size, spanish_vocab_size):
    model = Sequential([
        # Embedding layer using vocabulary size
        Embedding(input_dim=english_vocab_size + 2, output_dim=128, mask_zero=True),

        # SimpleRNN Layers
        SimpleRNN(128, return_sequences=True, dropout=0.1, activation='tanh'),
        Dropout(0.2),
        LayerNormalization(),

        SimpleRNN(256, return_sequences=True, dropout=0.1, activation='tanh'),
        Dropout(0.2),
        LayerNormalization(),

        # Dense Layers
        TimeDistributed(Dense(64, activation='relu')),
        Dropout(0.2),

        TimeDistributed(Dense(spanish_vocab_size + 2, activation='softmax'))
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.25, beta_1=0.9, beta_2=0.999)

# Question 5: Use the sparse categorical cross-entropy loss function
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
    )
    return model

# Define logits_to_text function
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = "<PAD>"

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('Input English sentences shape:', preproc_english_sentences.shape)
tmp_x = pad(preproc_english_sentences, max_spanish_length)
print('Padded input shape:', tmp_x.shape)
print('Target Spanish sentences shape:', preproc_spanish_sentences.shape)

# Train the RNN model
model = simple_translation_model(
    tmp_x.shape,
    max_spanish_length,
    english_vocab_size,
    spanish_vocab_size)
model.summary()

# Create Callbacks for Early Stopping & Model Checking
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )
]

# Question 6: Train the model for at least 5 epochs
history = model.fit(tmp_x, preproc_spanish_sentences,
                    batch_size=batch_size, epochs=epochs,
                    validation_split=0.2, callbacks=callbacks)

# Question 7: Plot the loss and accuracy curves for the train and validation sets
# Train and Validation Loss Curves
plt.figure(figsize=(6, 6))
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Train and Validation Accuracy Curves
plt.figure(figsize=(6, 6))
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

# Question 8: Prompt the user to enter an English sentence and translate to Spanish.
# Get English Sentence Input from User
user_sentence = input("Please enter an English sentence to translate: ").strip()
print("English Sentence: ", user_sentence)

# Translate English Sentence to Spanish
user_sentence = [english_tokenizer.word_index[word] for word in user_sentence.split()]
user_sentence = pad_sequences([user_sentence],
                              maxlen=preproc_spanish_sentences.shape[-2], padding='post')
user_sentence = user_sentence.reshape((-1, preproc_spanish_sentences.shape[-2], 1))

# Print Spanish Translation of English Sentence
prediction = model.predict(user_sentence)
print("Spanish Translation: ", logits_to_text(prediction[0], spanish_tokenizer))
