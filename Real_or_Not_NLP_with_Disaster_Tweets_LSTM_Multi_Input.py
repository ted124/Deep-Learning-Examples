import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import preprocessing, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Input, concatenate, Flatten

# path to data and other settings
path = "D:\\data\\Real_or_Not_NLP_with_Disaster_Tweets"
max_words = 20000
max_words_keyword = 500
maxlen = 150
maxlen_keyword = 5
embedding_dim = 300

# load data
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))

# fill na keyword with unknown
train_df['keyword'].fillna('unknown', inplace=True)
test_df['keyword'].fillna('unknown', inplace=True)

# split by %20
train_df['keyword'] = train_df['keyword'].str.split("%20").str.join(" ")
test_df['keyword'] = test_df['keyword'].str.split("%20").str.join(" ")

# create a tokenizer
tokenizer_text = keras.preprocessing.text.Tokenizer(
    num_words=max_words,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    char_level=False,
    oov_token=None,
    document_count=0
)
tokenizer_text.fit_on_texts(pd.concat([train_df, test_df], axis=0).reset_index(drop=True)["text"])
word_index_text = tokenizer_text.word_index

# process raw data
x_train = preprocessing.sequence.pad_sequences(tokenizer_text.texts_to_sequences(train_df['text']), maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(tokenizer_text.texts_to_sequences(test_df['text']), maxlen=maxlen)

# create a tokenizer for keyword
tokenizer_keyword = keras.preprocessing.text.Tokenizer(
    num_words=max_words_keyword,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    char_level=False,
    oov_token=None,
    document_count=0
)
tokenizer_keyword.fit_on_texts(pd.concat([train_df, test_df], axis=0).reset_index(drop=True)["keyword"])
word_index_keyword = tokenizer_keyword.word_index

# process raw data
x_train_keyword = preprocessing.sequence.pad_sequences(tokenizer_keyword.texts_to_sequences(train_df['keyword']), maxlen=maxlen_keyword)
x_test_keyword = preprocessing.sequence.pad_sequences(tokenizer_keyword.texts_to_sequences(test_df['keyword']), maxlen=maxlen_keyword)

# pretrained embedding_matrix
embedding_matrix = np.zeros((max_words, embedding_dim))

# pretrained embedding_matrix
embedding_matrix_keyword = np.zeros((max_words_keyword, embedding_dim))

# open file
f = open(os.path.join(path, 'glove.6B.300d.txt'), 'rb')

# generate pretrained embedding_matrix
for line in f:
    values = line.split()
    word = values[0].decode('utf-8')
    coefs = np.asarray(values[1:], dtype='float32')
    if word in word_index_text:
        if word_index_text[word] < max_words:
            embedding_matrix[word_index_text[word]] = coefs

    if word in word_index_keyword:
        if word_index_keyword[word] < max_words_keyword:
            embedding_matrix_keyword[word_index_keyword[word]] = coefs

# close file
f.close()

# create model
input_text = Input(shape=maxlen)
input_keyword = Input(shape=maxlen_keyword)

# embedding layers
embedding_text = Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=True)(input_text)
embedding_keyword = Embedding(max_words_keyword, embedding_dim, weights=[embedding_matrix_keyword], trainable=True)(input_keyword)

# lstm layers
lstm_text = LSTM(128, recurrent_dropout=0.01)(embedding_text)
lstm_keyword = LSTM(128, recurrent_dropout=0.01)(embedding_keyword)

# concate
lstm = concatenate([lstm_text, lstm_keyword])

# flatten
flatten = Flatten()(lstm)

# dropout
dropout_1 = Dropout(0.01)(flatten)

# dense
dense = Dense(128, activation='sigmoid')(dropout_1)

# dropout
dropout_2 = Dropout(0.01)(dense)

# output
output = Dense(1, activation='sigmoid')(dropout_2)

# create model
model = Model(inputs=[input_text, input_keyword], outputs=[output])

# compile
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
model.summary()

# early stopping
es_callback = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8, restore_best_weights=True)

# fit
history = model.fit([x_train, x_train_keyword],
                    train_df['target'],
                    batch_size=1024,
                    epochs=50,
                    validation_split=0.2,
                    callbacks=[es_callback])

# plot training history
training_loss = history.history['loss']
training_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

epochs = range(1, len(training_loss) + 1)
plt.figure()
plt.plot(epochs, training_loss, 'b--', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, training_acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# test
test_df["target"] = np.where(model.predict([x_test, x_test_keyword]) < 0.5, 0, 1)
result = test_df[["id", "target"]]

# to csv
result.to_csv(os.path.join(path, "result.csv"), index=False)