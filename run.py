import numpy as np
from common.common_functions import read_image_as_vector

from hopfield import HopfieldNetwork
from drawing import plot
from ui.board import *

if __name__ == "__main__":
#     seed = 1
#     np.random.seed(seed)
#     data = read_data_as_vectors('./data/sets/small-7x7.csv')
#     # data = read_data_as_vectors('./data/large-25x25.csv')

#     # train_data = data[:-1]
#     # test_data = data[-1]
#     train_data = data[:]
#     test_data = []

#     for pattern in train_data:
#         test_data.append(disturb_data(pattern,5))

#     train_data = np.asarray(train_data)
#     test_data = np.asarray(test_data)

#     nn = HopfieldNetwork(49)
#     # nn.train_oja(train_data,175, 0.0001)
#     nn.train(train_data)
#     res = []
#     for pattern in test_data:
#         res.append(nn.predict(pattern, 10, use_async=False, async_iter=10))
#     # plot(data, test_data, res)
#     board = Board(100, 7)
#     board.overwrite_board(data_to_array(res[0], (7, 7)))
#     board.run() # UNCOMMENT TO SHOW INTERACTIVE BOARD WITH THE FIRST RESULT

    data = read_image_as_vector('./data/big_set/ptak1.jpeg', resize=(80, 60))

    nn = HopfieldNetwork(len(data))
    nn.train([data])

    res = []
    res.append(nn.predict(data, 10))
    plot(data, data, res, size=(60,80))