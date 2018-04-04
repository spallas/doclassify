import os

import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding

from preprocessing import load_dataset


TRAIN_DIR = "../dataset/DATA/TRAIN"
VALID_DIR = "../dataset/DATA/DEV"

EMBEDDING_SIZE = 200
VOCABULARY_SIZE = 50_000
MAX_DOC_LEN = 1000


def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    print("INFO: Installed version: " + str(tf.VERSION))
    print("INFO: GPU found: ", tf.test.gpu_device_name())


def main(_):
    installation_test()

    classes_strings = sorted(os.listdir(TRAIN_DIR))
    class2int = {}
    for class_string in classes_strings:
        class2int[class_string] = len(class2int)

    x_train, y_train, x_devel, y_devel, embeddings = load_dataset(MAX_DOC_LEN)

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embeddings],
                                input_length=MAX_DOC_LEN,
                                trainable=False)

    print('Training model.')

    # train a 1D conv net with global max pooling
    sequence_input = Input(shape=(MAX_DOC_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    predicted = Dense(len(class2int), activation='softmax')(x)

    model = Model(sequence_input, predicted)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_devel, y_devel))


if __name__ == "__main__":
    tf.app.run()
