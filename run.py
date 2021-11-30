import numpy as np
from common.common_functions import read_image_as_vector, read_data_as_vectors, disturb_data

from hopfield import HopfieldNetwork
from drawing import plot
from ui.board import *

if __name__ == "__main__":
    STANDARD = True # DETERMINES IF THE IMAGES (False) OR STANDARD DATASETS (True) SHOULD BE PROCESSED
    USE_ASYNC = False # DETERMINES IF ASYNC PREDICT SHOULD BE USED
    DISTURBED = 10 # NUMBER OF PIXELS THAT SHOULD BE DISTURBED IN TEST IMAGES
    ITER = 100 # NUMBER OF ITERATIONS IN PREDICT
    ASYNC_ITER = 1000 # NUMBER OF ITERATIONS IN ASYNC PREDICT
    PATHS = ['./data/sets/small-7x7.csv', './data/sets/large-25x25.csv','./data/big_set/ptak1-266x200.jpg', './data/sets/small-7x7-modified1.csv', './data/sets/small-7x7-modified2.csv', './data/sets/small-7x7-modified4.csv'] # PATHS TO IMAGES
    NEURONS = [49, 625, None, 49, 49, 49] # NUMBER OF NEURONS FOR IMAGES ( SHOULD BE SET APPROPRIATELY TO CORRESPONDING PATH IN PATHS)
    IDX=5 # INDEX OF ELEMENTS IN PATHS AND NEURONS THAT WILL BE USED
    STANDARD_TO_OMMIT = [] # INDICES OF ELEMENTS THAT SHOULD NOT BE TAKEN INTO ACCOUNT
    OJA_LEARNING = False # DETERMINES IF OJA LEARNING SHOULD BE USED
    OJA_ITER = 100 # NUMBER OF ITERATIONS IN OJA LEARNING
    OJA_N = 0.001 # LEARNING COEFFICIENT IN OJA LEARNING
    if STANDARD:
        seed = 1
        ommit = STANDARD_TO_OMMIT
        np.random.seed(seed)
        data = read_data_as_vectors(PATHS[IDX])
        ommit.sort(reverse=True)
        for idx in ommit:
            del data[idx]
        train_data = data[:]
        test_data = []

        for pattern in train_data:
            test_data.append(disturb_data(pattern,DISTURBED))

        train_data = np.asarray(train_data)
        test_data = np.asarray(test_data)

        nn = HopfieldNetwork(NEURONS[IDX])
        nn.train_dataset(train_data)
        if OJA_LEARNING:
            nn.train_oja(train_data,OJA_ITER, OJA_N)

        res = []
        for pattern in test_data:
            res.append(nn.predict(pattern, ITER, use_async=USE_ASYNC, async_iter=ASYNC_ITER))
        plot(train_data, test_data, res)
#     board = Board(100, 7)
#     board.overwrite_board(data_to_array(res[0], (7, 7)))
#     board.run() # UNCOMMENT TO SHOW INTERACTIVE BOARD WITH THE FIRST RESULT
    else:
        data = read_image_as_vector(PATHS[IDX])

        nn = HopfieldNetwork(len(data))
        nn.train_dataset([data])
        if OJA_LEARNING:
            nn.train_oja([data],OJA_ITER, OJA_N)

        res = []
        test = disturb_data(data, DISTURBED)
        res.append(nn.predict(test, ITER, use_async=USE_ASYNC, async_iter=ASYNC_ITER))
        plot(data, test, res, size=(60,80))