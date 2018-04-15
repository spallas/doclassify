import collections
import os
import random
import re
import sys

import numpy as np
import pickle


from collections import Counter

STOP_WORDS_FILE = "./stopwords.txt"
# mostly retrieved at: https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords

VECTORS_FILES_NAME = "embeddings35M.pkl"
END_OF_DOC = "EODOC"
END_OF_DOMAIN = "EODOMAIN"
END_OF_TRAIN = "EOTRAIN"

context_center = 0
cross_batch = []


def generate_batch(batch_size, step, window_size, data):
    """
    This function generates the train data and label batch from the dataset considering
    samples of the type < window_size data[context_center] window_size >
    :param batch_size: the number of train_data,label pairs to produce per batch
    :param step: iteration, unused, for debugging purposes only
    :param window_size: how many words in the context
    :param data: the dataset
    :return: batch: train data for current batch
    :return: labels: labels for current batch
    """
    global context_center
    global cross_batch
    num_samples = 4
    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    context_start = max(0, context_center - window_size)
    context_end = min(len(data), context_center + window_size)
    # crop list to nearest end of doc.
    if END_OF_DOC in data[context_start:context_center]:
        context_start = len(data[context_start:context_center]) - \
                 data[context_start:context_center][::-1].index(END_OF_DOC) - 1
        # r index
    if END_OF_DOC in data[context_center:context_end]:
        context_end = data[context_center:context_end].index(END_OF_DOC)
    num_s = min(len(range(context_start, context_end)), num_samples)
    for index in random.sample(range(context_start, context_end), num_s):
        if index != context_center:
            cross_batch.append((data[context_center], data[index]))
    for i in range(batch_size):
        if len(cross_batch) is 0:
            context_center = (context_center + 1) % len(data)
            while data[context_center] == END_OF_DOC:
                context_center = (context_center + 1) % len(data)
            context_start = max(0, context_center - window_size)
            context_end = min(len(data), context_center + window_size)
            # crop list to nearest end of doc.
            if END_OF_DOC in data[context_start:context_center]:
                context_start = len(data[context_start:context_center]) - \
                                data[context_start:context_center][::-1].index(END_OF_DOC) - 1
                # r index
            if END_OF_DOC in data[context_center:context_end]:
                context_end = data[context_center:context_end].index(END_OF_DOC)
            num_s = min(len(range(context_start, context_end)), num_samples)
            for index in random.sample(range(context_start, context_end), num_s):
                if index != context_center:
                    cross_batch.append((data[context_center], data[index]))
        data_example = cross_batch.pop()
        batch[i] = data_example[0]
        labels[i] = data_example[1]
    context_center = (context_center + 1) % len(data)
    while data[context_center] == END_OF_DOC:
        context_center = (context_center + 1) % len(data)
    return batch, labels


def build_dataset(raw_word_list, vocab_size):
    """
    Transform raw word list in list of one-hot encoded (actually indexes
    in the range of the vocabulary size) list of words. Generates also
    the dictionaries to retrieve the original text.
    :param raw_word_list: a list of words
    :param vocab_size: the chosen size of vocabulary
    :return: data_indexes: list of codes (integers from 0 to vocabulary_size-1).
                This is the original text but words are replaced by their codes.
    :return: word2int: map of words(strings) to their codes(integers)
    :return: int2word: maps codes(integers) to words(strings)
    """
    data_indexes = []
    word2int = {}
    int2word = {}

    c = [("UNK", 0)]
    counter = collections.Counter(raw_word_list)
    c += counter.most_common(vocab_size - 3)
    print("INFO: ", list(enumerate(counter.most_common(20))))
    for i in enumerate(c):
        int2word[i[0]] = i[1][0]
        word2int[i[1][0]] = i[0]

    try:
        _ = word2int[END_OF_DOC]
        # could be already present in most common vocab_size words
    except KeyError:
        word2int[END_OF_DOC] = len(int2word)
        int2word[len(int2word)] = END_OF_DOC
    # manually add special symbols
    word2int[END_OF_DOMAIN] = len(int2word)
    int2word[len(int2word)] = END_OF_DOMAIN
    word2int[END_OF_TRAIN] = len(int2word)
    int2word[len(int2word)] = END_OF_TRAIN

    for token in raw_word_list:
        word_index = word2int.get(token, 0)
        # default=0 takes care of putting 0 in place
        # of less frequent words
        assert word_index >= 50_000
        data_indexes.append(word_index)

    assert len(word2int) != vocab_size

    return data_indexes, word2int, int2word


def read_and_preprocess(directory, domain_words=-1):
    """
    Preprocess text files
    :param directory: a directory containing text files organized in depth one subdirectories.
    :param domain_words: numbers of words to load for each subdirectory
    :return: raw_tokens_list: a list of preprocessed words
    :return: raw_pairs_list: list of all pairs of subsequent words from raw tokens list
    """
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

    raw_tokens_list = []
    raw_pairs_list = []
    stopwords = []
    with open(STOP_WORDS_FILE) as f:
        for word in f:
            stopwords.append(word.strip())
        stopwords = set(stopwords)
    print("LOG: reading directories: ", end="")
    sys.stdout.flush()
    for domain in sorted(os.listdir(directory)):
        domain_start = len(raw_tokens_list)
        for f_name in sorted(os.listdir(os.path.join(directory, domain))):
            if f_name.endswith(".txt"):
                with open(os.path.join(directory, domain, f_name)) as text_file:
                    for line in text_file.readlines():
                        line_tokens, line_pairs = _basic_preprocess(line, stopwords)
                        raw_tokens_list += line_tokens
                        raw_pairs_list += line_pairs
                    if len(raw_tokens_list) > domain_start + domain_words \
                            and domain_words > 0:
                        # crop
                        raw_tokens_list = raw_tokens_list[:domain_start + domain_words]
                        raw_tokens_list.append(END_OF_DOC)
                        # go to next domain
                        break
                raw_tokens_list.append(END_OF_DOC)
        raw_tokens_list.append(END_OF_DOMAIN)
        print("#", end="")
        sys.stdout.flush()
    print("\nLOG: Done.")
    raw_tokens_list.append(END_OF_TRAIN)
    return raw_tokens_list, raw_pairs_list


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
    sys.stdout.flush()
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


def save_vectors(vectors):
    """
    Simple pickle dump of python list
    :param vectors: python list
    """
    with open(VECTORS_FILES_NAME, "wb") as f:
        pickle.dump(vectors, f)


def read_analogies(file, dictionary):
    """
    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    :param file:
    :param dictionary:
    :return:
    """
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
