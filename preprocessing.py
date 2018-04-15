import os
import re
import sys

import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

END_OF_DOC = "EODOC"
END_OF_DOMAIN = "EODOMAIN"
END_OF_TRAIN = "EOTRAIN"

DATASET_FILE = "dataset35M.pkl"
EMBEDDINGS_FILE = "embeddings35M.pkl"
PREPROCESS_FILE = "Xy.pkl"
TESTS_FILE = "tests.pkl"

TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
STOP_WORDS_FILE = "stopwords.txt"


def load_dataset(max_doc_len):
    x_train = []
    y_train = []
    x_devel = []
    y_devel = []

    with open(DATASET_FILE, "rb") as f:
        data_onehot, word2int, int2word, _ = pickle.load(f)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_list = pickle.load(f)

    class_index = 0
    doc = []
    in_train = True
    domain_count = 0

    domains_dict = {}
    for word in data_onehot:
        if int2word[word] == END_OF_DOC:
            if in_train:
                x_train.append(doc)
                y_train.append(class_index)
            else:
                x_devel.append(doc)
                y_devel.append(class_index)
            doc = []
            domain_count += 1
            continue
        if int2word[word] == END_OF_TRAIN:
            in_train = False
            class_index = 0
            continue
        if int2word[word] == END_OF_DOMAIN:
            domains_dict[class_index] = domain_count / 79124  # VERY BAD, magic number...
            class_index += 1
            domain_count = 0
            continue
        doc.append(word)
        # Keras' embedding layer will apply the lookup.

    x_train = pad_sequences(x_train, maxlen=max_doc_len, dtype="float32", padding="post", truncating="post")
    y_train = np.asarray(y_train)
    x_devel = pad_sequences(x_devel, maxlen=max_doc_len, dtype="float32", padding="post", truncating="post")
    y_devel = np.asarray(y_devel)
    return x_train, y_train, x_devel, y_devel, np.asarray(embeddings_list), domains_dict


def load_for_sklearn():
    x_train = []
    y_train = []
    x_devel = []
    y_devel = []

    if os.path.exists("svmXy.pkl"):
        with open("svmXy.pkl", "rb") as f:
            x_train, y_train, x_devel, y_devel = pickle.load(f)

            print(np.shape(x_train), np.shape(y_train))

            return x_train, y_train, x_devel, y_devel, None, None

    with open(DATASET_FILE, "rb") as f:
        data_onehot, word2int, int2word, questions = pickle.load(f)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_list = pickle.load(f)

    in_train = True
    doc = []
    class_index = 0
    for word in data_onehot:
        if int2word[word] == END_OF_DOC:
            if len(doc) is 0:
                doc.append([0.0]*200)
            mean = np.asarray(doc).mean(axis=0).tolist()
            if in_train:
                x_train.append(mean)
                y_train.append(class_index)
            else:
                x_devel.append(mean)
                y_devel.append(class_index)
            doc = []
            continue
        if int2word[word] == END_OF_TRAIN:
            in_train = False
            class_index = 0
            continue
        if int2word[word] == END_OF_DOMAIN:
            class_index += 1
            continue
        doc.append(embeddings_list[word])

    x_train = np.asarray(x_train, dtype=np.float32)
    x_devel = np.asarray(x_devel, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_devel = np.asarray(y_devel, dtype=np.int32)

    print(np.shape(x_train))

    with open("svmXy.pkl", "wb") as f:
        pickle.dump([x_train, y_train, x_devel, y_devel], f)

    return x_train, y_train, x_devel, y_devel, None, None


def load_tests(max_doc_len, int2word):

    x_test = []

    with open(TESTS_FILE, "rb") as f:
        data_onehot, f_names_list = pickle.load(f)
    doc = []

    for word in data_onehot:
        if int2word[word] == END_OF_DOC:
            x_test.append(doc)
            doc = []
            continue
        if int2word[word] == END_OF_DOMAIN:
            continue
        doc.append(word)

    x_test = pad_sequences(x_test, maxlen=max_doc_len, dtype="float32", padding="post", truncating="post")

    return x_test, f_names_list
