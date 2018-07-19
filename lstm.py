from __future__ import print_function
import numpy as np
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.layers import Convolution1D
from keras.layers.recurrent import LSTM

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25



#max_features = 200000
filter_length = 5
nb_filter = 64
#pool_length = 4


if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
    q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
    q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
    labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
    word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    with open(NB_WORDS_DATA_FILE, 'r') as f:
        nb_words = json.load(f)['nb_words']
else:
    if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
        get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []
    is_duplicate = []
    with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(row['question1'])
            question2.append(row['question2'])
            is_duplicate.append(row['is_duplicate'])

    print('Question pairs: %d' % len(question1))

    questions = question1 + question2
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
        zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
        zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)

    np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
    np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
    np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)
X = np.stack((q1_data, q2_data), axis=1)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

'''Q1 = Sequential()
Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))


Q2 = Sequential()
Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))


model2 = Sequential()
model2.add(Merge([Q1, Q2], mode='concat'))
model2.add(BatchNormalization())
model2.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))
model3.add(GlobalMaxPooling1D())

model2.add(BatchNormalization())
model2.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))

model2.add(BatchNormalization())
model2.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))

model2.add(BatchNormalization())
model2.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))

model2.add(BatchNormalization())
'''

print ("1")

Q3 = Sequential()
Q3.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, dropout=0.2))


Q4 = Sequential()
Q4.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, dropout=0.2))

model1 = Sequential()
model1.add(Merge([Q3, Q4], mode='concat'))
model1.add(BatchNormalization())
model1.add(LSTM(200, dropout_W=0.2, dropout_U=0.2,return_sequences=True))
model1.add(BatchNormalization())
model1.add(LSTM(200, dropout_W=0.2, dropout_U=0.2,return_sequences=True))
model1.add(BatchNormalization())
model1.add(LSTM(200, dropout_W=0.2, dropout_U=0.2,return_sequences=True))
model1.add(BatchNormalization())
model1.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
model1.add(BatchNormalization())

'''model = Sequential()
model.add(Merge([model1, model2], mode='concat'))
model.add(BatchNormalization())
'''

model1.add(Dense(200, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(200, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(200, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(200, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(1, activation='sigmoid'))

print ("12")
model1.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])
print ("13")
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]

print("Starting training at", datetime.datetime.now())

t0 = time.time()
history = model1.fit([Q1_train, Q2_train], 
                    y_train, 
                    nb_epoch=NB_EPOCHS, 
                    validation_split=VALIDATION_SPLIT, 
                    verbose=1, 
                    callbacks=callbacks)
t1 = time.time()

print("Training ended at", datetime.datetime.now())

print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

model1.load_weights(MODEL_WEIGHTS_FILE)

loss, accuracy, precision, recall, fbeta_score = model1.evaluate([Q1_test, Q2_test], y_test)
print('')
print('loss      = {0:.4f}'.format(loss))
print('accuracy  = {0:.4f}'.format(accuracy))
print('precision = {0:.4f}'.format(precision))
print('recall    = {0:.4f}'.format(recall))
print('F         = {0:.4f}'.format(fbeta_score))
