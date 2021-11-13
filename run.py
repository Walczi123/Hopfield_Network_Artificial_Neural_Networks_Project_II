import numpy as np

from common.common_functions import array_to_vector, data_to_array, read_data_as_arrays, read_data_as_vectors
from hopfield import HopfieldNetwork
from drawing import plot
from ui.board import *

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
    board = Board(100, 7)
    board.overwrite_board(data_to_array(res, (7, 7)))
    board.run()
