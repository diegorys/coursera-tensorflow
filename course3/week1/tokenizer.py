import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print('Word index')
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
print('Sequences')
print(sequences)
padded = pad_sequences(sequences, padding='post')
print('Padded')
print(padded)
print(padded.shape)