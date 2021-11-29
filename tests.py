import multiprocessing
import os
import time
import tqdm
import glob
from pathlib import Path
from common.common_functions import read_data_as_vectors

from common.test.test import Test
from hopfield import HopfieldNetwork

PATH_TO_DATASET='./data/sets/'
PATH_TO_RESULTS='./results/'

def generate_instances():
    result = []

    for filepath in glob.glob(PATH_TO_DATASET+'*.csv'):
        data = read_data_as_vectors(filepath)
        nn = HopfieldNetwork(len(data[0]))
        for i in range(10):
            result.append(Test(nn,data,Path(filepath).stem,0.1 * i, 0.000001, 100, 100,100, PATH_TO_RESULTS))
    return result

def run_test(test):
    print(f'start of {test.name}')
    test.start_all()


def run_tests():
    iterable = generate_instances()
    # print(f'start of {iterable[0].name}')
    # iterable[0].start()

    start_time = time.time()

    max_cpu = multiprocessing.cpu_count()
    p = multiprocessing.Pool(int(max_cpu/2))
    for _ in tqdm.tqdm(p.imap_unordered(run_test, iterable), total=len(iterable)):
        pass
    # p.map_async(run_test, iterable)

    p.close()
    p.join()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()