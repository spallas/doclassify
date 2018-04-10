import numpy as np
import matplotlib.pyplot as plt


def plot_heat_matrix(confusion_matrix, classes_strings):
    norm_mat = []
    for line in confusion_matrix:
        line_norm = []
        for j in line:
            line_norm.append(float(j)/float(sum(line, 0)))
        norm_mat.append(line_norm)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_mat),
                    cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = confusion_matrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(confusion_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)

    plt.xticks(range(width), classes_strings)
    plt.yticks(range(height), classes_strings)
    plt.savefig('confusion_matrix.png', format='png')
