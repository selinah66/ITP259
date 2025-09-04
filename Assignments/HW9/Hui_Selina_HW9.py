# Selina Hui
# ITP 259 Fall 2024
# HW#9

import collections

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout
from matplotlib import pyplot as plt

from Homework import helper

# Constants
BATCH_SIZE = 300
EPOCHS = 25
GRU_UNITS = 256
DROPOUT_RATE = 0.3
VALIDATION_SPLIT = 0.2

# Load English data
english_sentences = helper.load_data('/Lecture18_19_RNN_Applications/small_vocab_en')
# Load French data
french_sentences = helper.load_data('/Lecture18_19_RNN_Applications/small_vocab_fr')

# Print sample translations
for sample_i in range(5):
    print('English sample {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('French sample {}:  {}\n'.format(sample_i + 1, french_sentences[sample_i]))

# Count vocabulary frequency for English and French, splits string into a list
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print(english_words_counter)
print(french_words_counter)

# Print Language Observations
print('\n{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('\n10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')

print('\n{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('\n10 Most common words in the French dataset:')
print('\n"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

# Define Tokenizer Function
def create_tokenizer(text, num_words=None):
    text_tokenizer = Tokenizer(
        num_words=num_words,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        oov_token='<UNK>'
    )
    text_tokenizer.fit_on_texts(text)
    return text_tokenizer.texts_to_sequences(text), text_tokenizer

# Define a Data Augmentation Function
def augment_data(sentences):
    augmented = []
    for sent in sentences:
        augmented.append(sent)
        # Simple augmentation to remove punctuation
        augmented.append(sent.replace('.', '').replace(',', ''))
    return augmented

# Define function to pad sentences to a given length
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

# Tokenize Sample Output
text_sentences = [
     "The quick brown fox jumps over the lazy dog .",
     "By Jove , my quick study of lexicography won a prize .",
     "This is a short sentence ."]
text_tokenized, text_tokenizer = create_tokenizer(text_sentences)
print('\n', text_tokenized)
print('\n', text_tokenizer.word_index)

# Pad Tokenized output. Default padding value is 0.
test_pad = pad(text_tokenized)

# Define Preprocessing Function
def preprocess(sentence_x, sentence_y):
    preprocess_x, x_tk = create_tokenizer(sentence_x)
    preprocess_y, y_tk = create_tokenizer(sentence_y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, \
french_tokenizer = preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

# Print tokens for each sentence
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
     print('Sequence {} in x'.format(sample_i + 1))
     print('  Input:  {}'.format(sent))
     print('  Output: {}'.format(token_sent))

for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))

# Define Recurrent Neural Network model and building layers
def simple_translation_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
        Build and train a basic RNN on x and y
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """
    model = Sequential([
        # Embedding layer using vocabulary size
        Embedding(input_dim=english_vocab_size + 1,
                  output_dim=256,
                  mask_zero=False),

        GRU(GRU_UNITS, return_sequences=True,
            dropout=DROPOUT_RATE,
            unroll=True),
        Dropout(DROPOUT_RATE),

        GRU(GRU_UNITS, return_sequences=True,
            dropout=DROPOUT_RATE,
            unroll=True),
        Dropout(DROPOUT_RATE),

        # Dense Layer with Time Distribution
        TimeDistributed(Dense(512, activation='relu')),
        Dropout(DROPOUT_RATE),
        TimeDistributed(Dense(french_vocab_size + 1, activation='softmax'))
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy']
    )
    return model

# Define logits_to_text function
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = "<PAD>"

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('Input English sentences shape:', preproc_english_sentences.shape)
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
print('Padded input shape:', tmp_x.shape)
print('Target French sentences shape:', preproc_french_sentences.shape)

# Train the RNN
model = simple_translation_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
model.summary()

# Callbacks for Early Stopping & Model Checking
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2
    )
]

history = model.fit(tmp_x, preproc_french_sentences,
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT, callbacks=callbacks)

# Question 1: Plot the Train and Validation Loss Curves
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Question 2: Plot the Train and Validation Accuracy Curves
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

# Question 3: Predict English Sentence
user_sentence = "she is driving a big green truck in paris and california"
print("English Sentence: ", user_sentence)

user_sentence = [english_tokenizer.word_index[word] for word in user_sentence.split()]
user_sentence = pad_sequences([user_sentence],
                              maxlen=preproc_french_sentences.shape[-2], padding='post')
user_sentence = user_sentence.reshape((-1, preproc_french_sentences.shape[-2], 1))

prediction = model.predict(user_sentence)
print("French Translation: ", logits_to_text(prediction[0], french_tokenizer))
