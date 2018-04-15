import os

import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, Dropout, LSTM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from collections import Counter

from preprocessing import load_dataset, load_for_sklearn
from evaluation import plot_heat_matrix


TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
MODE = "DNN"

EMBEDDING_SIZE = 200
VOCABULARY_SIZE = 50_000
MAX_DOC_LEN = 1024


def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    print("INFO: Installed version: " + str(tf.VERSION))
    print("INFO: GPU found: ", tf.test.gpu_device_name())


def main(_):
    """
    Please before running make sure all the files in the preprocessing code are present in the
    right location and with the right format.
    """
    installation_test()

    classes_strings = ['ANIMALS', 'ART_ARCHITECTURE_AND_ARCHAEOLOGY', 'BIOLOGY',
                       'BUSINESS_ECONOMICS_AND_FINANCE', 'CHEMISTRY_AND_MINERALOGY',
                       'COMPUTING', 'CULTURE_AND_SOCIETY', 'EDUCATION', 'ENGINEERING_AND_TECHNOLOGY',
                       'FARMING', 'FOOD_AND_DRINK', 'GAMES_AND_VIDEO_GAMES', 'GEOGRAPHY_AND_PLACES',
                       'GEOLOGY_AND_GEOPHYSICS', 'HEALTH_AND_MEDICINE', 'HERALDRY_HONORS_AND_VEXILLOLOGY',
                       'HISTORY', 'LANGUAGE_AND_LINGUISTICS', 'LAW_AND_CRIME', 'LITERATURE_AND_THEATRE',
                       'MATHEMATICS', 'MEDIA', 'METEOROLOGY', 'MUSIC', 'NUMISMATICS_AND_CURRENCIES',
                       'PHILOSOPHY_AND_PSYCHOLOGY', 'PHYSICS_AND_ASTRONOMY', 'POLITICS_AND_GOVERNMENT',
                       'RELIGION_MYSTICISM_AND_MYTHOLOGY', 'ROYALTY_AND_NOBILITY', 'SPORT_AND_RECREATION',
                       'TEXTILE_AND_CLOTHING', 'TRANSPORT_AND_TRAVEL', 'WARFARE_AND_DEFENSE']
    class2int = {}
    for class_string in classes_strings:
        class2int[class_string] = len(class2int)

    x_train, y_train, x_devel, y_devel, embeddings, domain_dict = load_dataset(MAX_DOC_LEN) \
        if MODE is "DNN" else load_for_sklearn()

    print('Training model.')

    if MODE is "SVM":
        svm_domain_dict = {}
        classes_counts = Counter(y_train.tolist())
        for c in classes_counts:
            svm_domain_dict[c] = (1 - classes_counts[c] / len(y_train))
        # manually adjust underrepresented classes
        svm_domain_dict[classes_strings.index("CULTURE_AND_SOCIETY")] = 9.0
        svm_domain_dict[classes_strings.index("METEOROLOGY")] = 5.0
        svm_domain_dict[classes_strings.index("HISTORY")] = 5.0
        svm_domain_dict[classes_strings.index("ENGINEERING_AND_TECHNOLOGY")] = 5.0
        x_weights = []
        for i in y_train:
            x_weights.append(svm_domain_dict[i])

        classifier = svm.LinearSVC()
        classifier.fit(x_train, y_train, sample_weight=x_weights)
        y_pred = classifier.predict(x_devel)

        print(classification_report(y_devel, y_pred, target_names=classes_strings))
        return

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embeddings],
                                input_length=MAX_DOC_LEN,
                                trainable=False)

    # train a 1D conv net with global max pooling
    input_layer = Input(shape=(MAX_DOC_LEN,), dtype='float32')
    embeddings_output = embedding_layer(input_layer)
    x = Conv1D(128, 5, activation='relu')(embeddings_output)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = LSTM(150, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction_layer = Dense(len(class2int), activation='softmax')(x)

    model = Model(input_layer, prediction_layer)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=256,
              epochs=22,
              validation_data=(x_devel, y_devel),
              class_weight=domain_dict)

    model.save("model.h5")

    y_prob = model.predict(x_devel)
    y_pred = y_prob.argmax(axis=-1)
    plot_heat_matrix(confusion_matrix(y_devel, y_pred))
    print(classification_report(y_devel, y_pred, target_names=classes_strings))


if __name__ == "__main__":
    tf.app.run()
