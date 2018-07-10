import os
import pickle

from keras.preprocessing.sequence import pad_sequences

from preprocessing import TESTS_FILE, DATASET_FILE, read_and_preprocess_tests
from doclassify import MAX_DOC_LEN
from keras.models import load_model


# Produce the output file for the competition.

with open(DATASET_FILE, "rb") as f:
    _, word2int, int2word, _ = pickle.load(f)

x_test = []  # tests padded lists sorted by test name.

if os.path.exists(TESTS_FILE):
    with open(TESTS_FILE, "rb") as f:
        tests_dict = pickle.load(f)
else:
    tests_dict = read_and_preprocess_tests()

for test_name in sorted(tests_dict):
    x_test.append(tests_dict[test_name])

x_test = pad_sequences(x_test, maxlen=MAX_DOC_LEN, dtype="float32", padding="post", truncating="post")

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

with open("test_answers.tsv", "w") as f:
    i = 0
    for test_name in sorted(tests_dict.keys()):
        print("{}\t{}".format(test_name[:-4], classes_strings[y_pred[i]]), file=f)
        i += 1
