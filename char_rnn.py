import os
import re
import sys

import keras.callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, concatenate, TimeDistributed, Bidirectional


maxlen = 512
max_sentences = 15
filter_length = [5, 3, 3]
nb_filter = [196, 196, 256]
pool_length = 2
char_embedding = 40
VALIDATION_SPLIT = 0.2


# record history of training
class RecordLossHistory(keras.callbacks.Callback):
    def __init__(self):
        super(RecordLossHistory, self).__init__()
        self.accuracies = []
        self.losses = []

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


# Model 2 with Convolutional and fully connected layers followed by LSTM encoder/decoder
def sentence_encoder_wo_dense(layer):
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
        layer = Conv1D(filters=nb_filter[i], kernel_size=filter_length[i], padding='valid', activation='relu',
                       kernel_initializer='glorot_normal', strides=1)(layer)

        layer = Dropout(0.1)(layer)
        layer = MaxPooling1D(pool_size=pool_length)(layer)
    bi_lstm_sent = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                                      implementation=0))(layer)
    sentence_encode = Dropout(0.2)(bi_lstm_sent)
    # sentence encoder
    encoder = Model(inputs=input_sentence, outputs=sentence_encode)
    encoder.summary()
    encoded = TimeDistributed(encoder)(document)
    # encoded: sentences to bi-lstm for document encoding
    bi_lstm_doc = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                                     implementation=0))(encoded)
    output = Dropout(0.2)(bi_lstm_doc)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.2)(output)
    output = Dense(1, activation='sigmoid')(output)
    return output


def char_block(in_layer, filters, filter_len, subsample, pool_len):
    block = in_layer
    for i in range(len(filters)):

        block = Conv1D(filters=filters[i], kernel_size=filter_len[i], padding='valid', activation='tanh',
                       strides=subsample[i])(block)
        # block = Dropout(0.1)(block)
        if pool_len[i]:
            block = MaxPooling1D(pool_size=pool_len[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


# Model 2 with Convolutional and fully connected layers followed by LSTM encoder/decoder
def sentence_encoder_with_dense(layer):
    block_2 = char_block(layer, filters=(128, 256), filter_len=(5, 5), subsample=(1, 1), pool_len=(2, 2))
    block_3 = char_block(layer, filters=(192, 320), filter_len=(7, 5), subsample=(1, 1), pool_len=(2, 2))

    sentence_encode = concatenate([block_2, block_3], axis=-1)
    # sent_encode = Dropout(0.2)(sent_encode)

    encoder = Model(inputs=input_sentence, outputs=sentence_encode)
    encoder.summary()

    encoded = TimeDistributed(encoder)(document)

    lstm_h = 92

    lstm_layer1 = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(
        lstm_layer1)

    # output = Dropout(0.2)(bi_lstm)
    output = Dense(1, activation='sigmoid')(lstm_layer2)
    return output


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def remove_html(str_a):
    p = re.compile(r'<.*?>')
    return p.sub('', str_a)


# replace all non-ASCII (\x00-\x7F) characters with a space
def replace_non_ascii(str_a):
    return re.sub(r'[^\x00-\x7f]', r'', str_a)


checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

input_data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
txt = ''
reviews = []
sentences = []
sentiments = []
num_sent = []

for rev, sentiment in zip(input_data.review, input_data.sentiment):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', replace_non_ascii(remove_html(rev)))
    sentences = [sent.lower() for sent in sentences]
    reviews.append(sentences)
    sentiments.append(sentiment)

for rev in reviews:
    num_sent.append(len(rev))
    for s in rev:
        txt += s

chars = set(txt)
max_features = len(chars) + 1
print('Total # of  chars in dataset:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.ones((len(reviews), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(sentiments)

for i, doc in enumerate(reviews):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):  # type: (int, object)
                X[i, j, (maxlen - 1 - t)] = char_indices[char]


# shuffle X and y
ids = np.arange(len(X))
np.random.shuffle(ids)
X = X[ids]
y = y[ids]
nb_validation_samples = int(VALIDATION_SPLIT * X.shape[0])

# Test/Train split 20K Train, 5K Test
X_train = X[:-nb_validation_samples]
y_train = y[:-nb_validation_samples]
X_val = X[-nb_validation_samples:]
y_val = y[-nb_validation_samples:]

# document input
document = Input(shape=(max_sentences, maxlen), dtype='int64')
# sentence input
input_sentence = Input(shape=(maxlen,), dtype='int64')

# char indices to one hot matrix, 1D sequence to 2D 
embedded_layer = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)

print('running model without fully connected layers')
model = Model(inputs=document, outputs=sentence_encoder_wo_dense(embedded_layer))
# model = Model(inputs=document, outputs=sentence_encoder_with_dense(embedded_layer))
model.summary()

if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]
ckpt_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss', verbose=0, save_best_only=True, mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

loss_history = RecordLossHistory()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=100, epochs=10, shuffle=True,
          callbacks=[earlystop_cb, ckpt_cb, loss_history])

# print loss_history.losses
# print loss_history.accuracies
