import collections
import os
import random
import re
import sys

import numpy as np
import pickle


from collections import Counter

STOP_WORDS_FILE = "stopwords.txt"
# mostly retrieved at: https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords

VECTORS_FILES_NAME = "embeddings35M.pkl"
END_OF_DOC = "EODOC"
END_OF_DOMAIN = "EODOMAIN"
END_OF_TRAIN = "EOTRAIN"


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
    f_names_list = []
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
                f_names_list.append(f_name)
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
    return raw_tokens_list, raw_pairs_list, f_names_list


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

