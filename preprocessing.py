
import numpy as np
import pickle


def load_dataset(max_doc_len):
    x_train = []
    # one element in this list is:
    # matrix for document:
    #   [[emb-val1, ..., emb-val200], [...], ...]
    #               word1      ,      word2, ...
    y_train = []
    x_devel = []
    y_devel = []

    with open("dataset.pkl", "rb") as f:
        data_onehot, word2int, int2word, questions = pickle.load(f)

    with open("embeddings.pkl", "rb") as f:
        embeddings_list = pickle.load(f)

    class_index = 0
    doc = []
    in_train = True
    for word in data_onehot:
        if int2word[word] == "EODOC":
            if in_train:
                x_train.append(doc)
                y_train.append(class_index)
            else:
                x_devel.append(doc)
                y_devel.append(class_index)
            doc = []
            continue
        if int2word[word] == "EOTRAIN":
            in_train = False
        if int2word[word] == "EODOMAIN":
            class_index += 1
            continue
        doc.append(word)
        # Keras' embedding layer will apply the lookup.

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_devel = np.asarray(x_devel)
    y_devel = np.asarray(y_devel)
    return x_train, y_train, x_devel, y_devel, np.asarray(embeddings_list)
