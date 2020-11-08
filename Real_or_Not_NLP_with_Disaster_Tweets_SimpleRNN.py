import os

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Dense, Embedding, Dropout, SimpleRNN
from tensorflow.keras.models import Sequential

# path to data and other settings
path = "D:\\data\\Real_or_Not_NLP_with_Disaster_Tweets"
max_words = 10000
maxlen = 150
embedding_dim = 300

# load data
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))

# create a tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=10000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    char_level=False,
    oov_token=None,
    document_count=0
)
tokenizer.fit_on_texts(train_df["text"])
word_index = tokenizer.word_index

# process raw data
x_train = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_df['text']), maxlen=maxlen)

# pretrained embedding_matrix
embedding_matrix = np.zeros((max_words, embedding_dim))

# open file
f = open(os.path.join(path, 'glove.6B.300d.txt'), 'rb')

# generate pretrained embedding_matrix
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    if word in word_index:
        if word_index[word] < max_words:
            embedding_matrix[word_index[word]] = coefs

# close file
f.close()

# create model
model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=maxlen))
model.add(Dropout(0.02))
model.add(SimpleRNN(64))
model.add(Dropout(0.02))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# set Embedding to not trainable
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# fit
history = model.fit(x_train,
                    train_df['target'],
                    batch_size=128,
                    epochs=10,
                    validation_split=0.2)

# test
test_df["target"] = np.where(model.predict(x_test) < 0.5, 0, 1)
result = test_df[["id", "target"]]
