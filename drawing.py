import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def plot(data, test, predicted, figsize=(5, 6)):
    data = np.atleast_2d(data)
    test = np.atleast_2d(test)
    predicted = np.atleast_2d(predicted)
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(max(len(data), len(test), len(predicted))):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
        if len(data) > i:
            axarr[i, 0].imshow(data[i], cmap='gray_r')
            axarr[i, 0].axis('off')
        if len(test) > i:
            axarr[i, 1].imshow(test[i], cmap='gray_r')
            axarr[i, 1].axis('off')
        if len(predicted) > i:
            axarr[i, 2].imshow(predicted[i], cmap='gray_r')
            axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()
