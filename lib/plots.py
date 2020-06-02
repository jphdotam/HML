import matplotlib.pyplot as plt
import numpy as np
import itertools

from matplotlib.colors import LinearSegmentedColormap

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

def plot_confusion_matrix(cm,
                          classes,
                          title,
                          xlab=r"$\bf{True}$" + ' beat class',
                          ylab=r"$\bf{Predicted}$" + ' beat class',
                          figsize=(8,8)):

    plt.rcParams["figure.figsize"] = figsize
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    cdict = {'red': [(0.0, 1, 1), (0.125, 0.8705882352941177, 0.8705882352941177),
                                 (0.25, 0.7764705882352941, 0.7764705882352941),
                                 (0.375, 0.6196078431372549, 0.6196078431372549),
                                 (0.5, 0.4196078431372549, 0.4196078431372549),
                                 (0.625, 0.25882352941176473, 0.25882352941176473),
                                 (0.75, 0.12941176470588237, 0.12941176470588237),
                                 (0.875, 0.03137254901960784, 0.03137254901960784),
                                 (1.0, 0.03137254901960784, 0.03137254901960784)],
                         'green': [(0.0, 1, 1), (0.125, 0.9215686274509803, 0.9215686274509803),
                                   (0.25, 0.8588235294117647, 0.8588235294117647),
                                   (0.375, 0.792156862745098, 0.792156862745098),
                                   (0.5, 0.6823529411764706, 0.6823529411764706),
                                   (0.625, 0.5725490196078431, 0.5725490196078431),
                                   (0.75, 0.44313725490196076, 0.44313725490196076),
                                   (0.875, 0.3176470588235294, 0.3176470588235294),
                                   (1.0, 0.18823529411764706, 0.18823529411764706)],
                         'blue': [(0.0, 1.0, 1.0), (0.125, 0.9686274509803922, 0.9686274509803922),
                                  (0.25, 0.9372549019607843, 0.9372549019607843),
                                  (0.375, 0.8823529411764706, 0.8823529411764706),
                                  (0.5, 0.8392156862745098, 0.8392156862745098),
                                  (0.625, 0.7764705882352941, 0.7764705882352941),
                                  (0.75, 0.7098039215686275, 0.7098039215686275),
                                  (0.875, 0.611764705882353, 0.611764705882353),
                                  (1.0, 0.4196078431372549, 0.4196078431372549)],
                         'alpha': [(0.0, 1, 1), (0.125, 1, 1), (0.25, 1, 1), (0.375, 1, 1), (0.5, 1, 1), (0.625, 1, 1),
                                   (0.75, 1, 1), (0.875, 1, 1), (1.0, 1, 1)]}

    cmap = LinearSegmentedColormap('cm_map',segmentdata=cdict, N=256)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")

    print(cm_norm)

    plt.figure(len(classes))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title, pad=25)
    cbar = plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    cbar.ax.set_ylabel('Proportion (%)', rotation=270, labelpad=25)

    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45, horizontalalignment="right")  # HA needed if rotated labels
    plt.xticks(tick_marks, classes)  # HA needed if rotated labels
    plt.yticks(tick_marks, classes)

    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        plt.text(j, i, f"{(cm[i, j])} ({cm_norm[i, j] * 100:.2f}%)",
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black")

    plt.subplots_adjust(bottom=0.3)
    plt.ylabel(xlab, labelpad=20)
    plt.xlabel(ylab, labelpad=20)
    plt.savefig("./cm.pdf")
    plt.show()

    return cm_norm