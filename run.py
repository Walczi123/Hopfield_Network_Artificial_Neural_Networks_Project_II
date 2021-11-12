import numpy as np

from common.common_functions import array_to_vector, data_to_array, read_data_as_arrays, read_data_as_vectors
from hopfield import HopfieldNetwork
from matplotlib import pyplot as plt

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    data = read_data_as_vectors('./data/small-7x7.csv')
    # print(data)

    train_data = data[:-1]
    test_data = data[-1]

    a = np.array([1,2,3])
    b = np.array([1,2,3])
    w = np.outer(a, b)

    diagW = np.diag(np.diag(w))
    # print(diagW)
    W = w - diagW
    # print(W)
    # W = W/3
    print(W.shape[0])