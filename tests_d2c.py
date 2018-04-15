import pickle

from preprocessing import TESTS_FILE, DATASET_FILE, load_tests
from data_preprocessing_d2c import read_and_preprocess, detect_phrases, build_pairs
from keras.models import load_model

TRAIN = False
LOAD = False
# Produce the output file for the competition.

if LOAD:
    with open(DATASET_FILE, "rb") as f:
        _, word2int, int2word, _ = pickle.load(f)

    raw_tokens_list, raw_pairs_list, f_names_list = read_and_preprocess("test")
    words_list = detect_phrases(raw_tokens_list, raw_pairs_list)
    words_list = detect_phrases(words_list, build_pairs(words_list))

    data = []
    for token in words_list:
        word_index = word2int.get(token, 0)
        data.append(word_index)

    with open(TESTS_FILE, "wb") as f:
        pickle.dump([data, f_names_list], f)

if TRAIN:
    with open(DATASET_FILE, "rb") as f:
        _, word2int, int2word, _ = pickle.load(f)

    x_test, names_list = load_tests(1024, int2word)

    model = load_model("model.h5")
    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)

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

    with open("predictions.tsv", "w") as f:
        i = 0
        for y in y_pred:
            print("{}\t{}".format(names_list[i], classes_strings[y]), file=f)
            i += 1


with open("predictions.tsv", "r") as f:
    with open("test_answers.tsv", "w") as f2:
        for line in f:
            id, cl = line.strip().split("\t")
            print("{}\t{}".format(id, cl), file=f2)
