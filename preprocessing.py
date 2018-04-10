
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

END_OF_DOC = "EODOC"
END_OF_DOMAIN = "EODOMAIN"
END_OF_TRAIN = "EOTRAIN"

DATASET_FILE = "../dataset2M.pkl"
EMBEDDINGS_FILE = "../embeddings2M33acc.pkl"


def load_dataset(max_doc_len):
    x_train = []
    y_train = []
    x_devel = []
    y_devel = []

    with open(DATASET_FILE, "rb") as f:
        data_onehot, word2int, int2word, questions = pickle.load(f)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_list = pickle.load(f)

    class_index = 0
    doc = []
    in_train = True
    for word in data_onehot:
        if int2word[word] == END_OF_DOC:
            if in_train:
                x_train.append(doc)
                y_train.append(class_index)
            else:
                x_devel.append(doc)
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
        doc.append(word)
        # Keras' embedding layer will apply the lookup.

    x_train = pad_sequences(x_train, maxlen=max_doc_len, dtype="float32", padding="post", truncating="post")
    y_train = np.asarray(y_train)
    x_devel = pad_sequences(x_devel, maxlen=max_doc_len, dtype="float32", padding="post", truncating="post")
    y_devel = np.asarray(y_devel)
    return x_train, y_train, x_devel, y_devel, np.asarray(embeddings_list)


def load_tests():
    pass
