import os
import re

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

TRAIN_DIR = "../dataset/DATA/TRAIN"
VALID_DIR = "../dataset/DATA/DEV"
TESTS_DIR = "../dataset/DATA/TEST"
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


def read_and_preprocess_tests(directory=TESTS_DIR):

    def _basic_preprocess(file_line, stopwords_set):
        """
        Filter punctuation, skip latex keywords and numbers,
        skip stopwords, prepare data for initial detecting of word pairs appearing
        very often.
        :param file_line:
        :param stopwords_set:
        :return:
        """
        tokens_split = []
        pairs_split = []
        prev_token = ""
        for token in re.split('[^a-zA-Z0-9_èéàìòù\\\\]+', file_line.strip().lower()):
            if len(token) == 0 or token in stopwords_set:
                continue
            if (not token.isalpha()) or (token[0] is "\\"):
                # then is number or latex expression
                continue
                # result.append("<number>")
            else:
                if prev_token is not "":
                    pairs_split.append(prev_token + "_" + token)
                tokens_split.append(token)
            prev_token = token
        return tokens_split, pairs_split

    # RETRIEVE STOPWORDS
    stopwords = []
    with open(STOP_WORDS_FILE) as f:
        for word in f:
            stopwords.append(word.strip())
        stopwords = set(stopwords)

    # LOAD DICTIONARIES FROM WORD2VEC
    with open(DATASET_FILE, "rb") as f:
        _, word2int, int2word, _ = pickle.load(f)

    tests_dict = {}  # key = test_name; value = list of one-hot words of doc

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            test_tokens = []
            for line in f:
                line_tokens, _ = _basic_preprocess(line, stopwords)
                test_tokens += line_tokens
            test_onehots = []
            for token in test_tokens:
                word_index = word2int.get(token, 0)  # 0 for unknown words
                test_onehots.append(word_index)

        tests_dict[filename] = test_onehots

    with open(TESTS_FILE, "wb") as f:
        pickle.dump(tests_dict, f)

    return tests_dict


def detect_phrases(raw_token_list, raw_pairs_list):
    """
    Detect pairs of words appearing frequently together:
    E.g.: new york -> new_york, video game -> video_game
    Adapted from:
    https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2phrase.c
    :param raw_pairs_list: list of words
    :param raw_token_list: list of word pairs obtained from first param
    :return: list of words and word pairs
    """
    print("LOG: detecting phrases...")
    min_count = 5
    threshold = 100
    token_counter = Counter(raw_token_list)
    pairs_counter = Counter(raw_pairs_list)

    last_word = raw_token_list[0]
    result_string = [last_word]
    curr_count = 0
    last_count = token_counter[last_word]
    for word in raw_token_list[1:]:
        oov = False  # out of vocabulary situation
        if word is "UNK":
            oov = True
        else:
            curr_count = token_counter[word]

        if last_word == "UNK" or last_word == END_OF_DOC \
                or last_word == END_OF_DOMAIN:
            oov = True
        bigram = last_word + "_" + word
        pair_count = pairs_counter[bigram]
        if pair_count is 0 \
                or curr_count < min_count \
                or last_count < min_count:
            oov = True
        if oov:
            score = 0.0
        else:
            score = (pair_count - min_count) / last_count / curr_count
            score *= len(raw_token_list)

        if score > threshold:
            # add both to output
            result_string.append("_" + word)
            curr_count = 0
        else:
            # add word alone
            result_string.append(" " + word)
            pass

        last_word = word
        last_count = curr_count

    result_string = "".join(result_string)
    return result_string.split()


def build_pairs(l):
    pairs = []
    last = l[0]
    for w in l:
        pairs.append(last + "_" + w)
        last = w
    return pairs
