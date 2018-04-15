
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model


classes_strings = ['ANIMALS', 'ART ARCHITECTURE AND ARCHAEOLOGY', 'BIOLOGY',
                   'BUSINESS ECONOMICS AND FINANCE', 'CHEMISTRY AND MINERALOGY',
                   'COMPUTING', 'CULTURE AND SOCIETY', 'EDUCATION', 'ENGINEERING AND TECHNOLOGY',
                   'FARMING', 'FOOD AND DRINK', 'GAMES AND VIDEO GAMES', 'GEOGRAPHY AND PLACES',
                   'GEOLOGY AND GEOPHYSICS', 'HEALTH AND MEDICINE', 'HERALDRY HONORS AND VEXILLOLOGY',
                   'HISTORY', 'LANGUAGE AND LINGUISTICS', 'LAW AND CRIME', 'LITERATURE AND THEATRE',
                   'MATHEMATICS', 'MEDIA', 'METEOROLOGY', 'MUSIC', 'NUMISMATICS AND CURRENCIES',
                   'PHILOSOPHY AND PSYCHOLOGY', 'PHYSICS AND ASTRONOMY', 'POLITICS AND GOVERNMENT',
                   'RELIGION MYSTICISM AND MYTHOLOGY', 'ROYALTY AND NOBILITY', 'SPORT AND RECREATION',
                   'TEXTILE AND CLOTHING', 'TRANSPORT AND TRAVEL', 'WARFARE AND DEFENSE']


def plot_heat_matrix(c_matrix):
    cm_dataframe = pd.DataFrame(c_matrix, index=classes_strings,
                                columns=classes_strings)

    plt.subplots(figsize=(10, 10))
    ax = sn.heatmap(cm_dataframe, cmap=plt.cm.jet, annot=False, square=True)
    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    plt.tight_layout()
    plt.show()


def evaluate_saved_model(path, x_test, y_test):

    model = load_model(path)
    print("Predicting...")
    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)

    norm_conf = []
    for line in confusion_matrix(y_test, y_pred):
        res_line = []
        total = sum(line)
        for j in line:
            res_line.append(j / total)
        norm_conf.append(res_line)

    df_cm = pd.DataFrame(norm_conf, index=classes_strings,
                         columns=classes_strings)

    plt.subplots(figsize=(14, 14))
    ax = sn.heatmap(df_cm, cmap=plt.cm.jet, annot=False, square=True)
    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    plt.tight_layout()
    plt.show()
    print(classification_report(y_test, y_pred, target_names=classes_strings))
