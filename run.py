import numpy as np

from common.common_functions import array_to_vector, data_to_array, read_data_as_arrays, read_data_as_vectors
from hopfield import HopfieldNetwork
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


if __name__ == "__main__":
    data = read_data_as_vectors('./data/small-7x7.csv')

    train_data = data[:-1]
    test_data = data[-1]
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    nn = HopfieldNetwork(49, 10)
    print('train_data: ', train_data)
    nn.train(train_data)

    res = nn.predict(test_data)
    print('test_data', data_to_array(test_data, (7, 7)))
    print('res', data_to_array(res, (7, 7)))
    plot(data, test_data, res)
