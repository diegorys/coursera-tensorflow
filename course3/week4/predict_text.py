import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

tokenizer = Tokenizer()

data = "In the town of Athy one Jeremy Lanigan \nbattered away till he hadn't a pound \nhis father he died and made him a man again \nleft a farm with ten acres of ground \nhe gave a grand party for friends a relations \nwho did not forget him when come to the will \nand if you'll but listen I'll make you're eyes glisten \nof rows and ructions at Lanigan's Ball \nsix long months I spent in Dub-i-lin \nsix long months doing nothin' at all \nsix long months I spent in Dub-i-lin \nlearning to dance for Lanigan's Ball \nI stepped out I stepped in again \nI stepped out I stepped in again \nI stepped out I stepped in again \nlearning to dance for Lanigan's Ball \nMyself to be sure got free invitaions \nfor all the nice boys and girls I did ask \nin less than 10 minutes the friends and relations \nwere dancing as merry as bee 'round a cask \nThere was lashing of punch and wine for the ladies \npotatoes and cakes there was bacon a tay \nthere were the O'Shaughnessys, Murphys, Walshes, O'Gradys \ncourtin' the girls and dancing away \nthey were doing all kinds of nonsensical polkas \nall 'round the room in a whirly gig \nbut Julia and I soon banished their nonsense \nand tipped them a twist of a real Irish jig \nOh how that girl got mad on me \nand danced till you'd think the ceilings would fall \nfor I spent three weeks at Brook's academy \nlearning to dance for Lanigan's Ball CHORUS \nThe boys were all merry the girls were all hearty \ndancing away in couples and groups \ntill an accident happened young Terrance McCarthy \nput his right leg through Miss Finerty's hoops \nThe creature she fainted and cried 'melia murder' \ncried for her brothers and gathered them all \nCarmody swore that he'd go no further \ntill he'd have satisfaction at Lanigan's Ball \nIn the midst of the row Miss Kerrigan fainted \nher cheeks at the same time as red as a rose \nsome of the boys decreed she was painted \nshe took a wee drop too much I suppose \nHer sweetheart Ned Morgan all powerful and able \nwhen he saw his fair colleen stretched out by the wall \nhe tore the left leg from under the table \nand smashed all the dishes at Lanigan's Ball CHORUS \nBoy oh Boys tis then there was ructions \nmyself got a kick from big Phelam McHugh \nbut soon I replied to this kind introduction \nand kicked up a terrible hullaballoo \nold Casey the piper was near being strangled \nthey squeezed up his pipes bellows chanters and all \nthe girls in their ribbons they all got entangled \nand that put an end to Lanigan's Ball CHORUS"

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
# +1 because of we are considering the OOV.
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)
    for i in range(1, len(token_list)):
        n_gram_secuence = token_list[:i+1]
        input_sequences.append(n_gram_secuence)
print('Input sequences')
print(input_sequences)
max_sequence_len = max([len(x) for x in input_sequences])
print('Max sequence length: ' + str(max_sequence_len))
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))
print('Padded input sequences')
print(input_sequences)

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

print('Input xs')
print(xs)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print('Labels')
print(labels)
print('Output ys')
print(ys)

print(tokenizer.word_index['in'])
print(tokenizer.word_index['the'])
print(tokenizer.word_index['town'])
print(tokenizer.word_index['of'])
print(tokenizer.word_index['athy'])
print(tokenizer.word_index['one'])
print(tokenizer.word_index['jeremy'])
print(tokenizer.word_index['lanigan'])


model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=500, verbose=1)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


plot_graphs(history, 'accuracy')

seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
