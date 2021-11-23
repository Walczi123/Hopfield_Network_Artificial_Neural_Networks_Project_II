import numpy as np
from common.common_functions import read_image_as_vector, read_data_as_vectors, disturb_data

from hopfield import HopfieldNetwork
from drawing import plot
from ui.board import *

if __name__ == "__main__":
    STANDARD = True
    USE_ASYNC = False
    if STANDARD:
        seed = 1
        ommit = [2] # INDICES OF ELEMENTS THAT SHOULD NOT BE TAKEN INTO ACCOUNT
        np.random.seed(seed)
        data = read_data_as_vectors('./data/sets/small-7x7.csv')
        # data = read_data_as_vectors('./data/large-25x25.csv')
        ommit.sort(reverse=True)
        for idx in ommit:
            del data[idx]
        # train_data = data[:-1]
        # test_data = data[-1]
        train_data = data[:]
        test_data = []

        for pattern in train_data:
            test_data.append(disturb_data(pattern,5))

        train_data = np.asarray(train_data)
        test_data = np.asarray(test_data)

        nn = HopfieldNetwork(49)
        # nn.train_oja(train_data,175, 0.0001)
        nn.train(train_data)
        res = []
        for pattern in test_data:
            res.append(nn.predict(pattern, 10, use_async=USE_ASYNC, async_iter=10))
        plot(data, test_data, res)
#     board = Board(100, 7)
#     board.overwrite_board(data_to_array(res[0], (7, 7)))
#     board.run() # UNCOMMENT TO SHOW INTERACTIVE BOARD WITH THE FIRST RESULT
    else:
        data = read_image_as_vector('./data/big_set/ptak1.jpeg', resize=(80, 60))

        nn = HopfieldNetwork(len(data))
        nn.train([data])

        res = []
        test = disturb_data(data, 100)
        res.append(nn.predict(test, 10))
        plot(data, test, res, size=(60,80))