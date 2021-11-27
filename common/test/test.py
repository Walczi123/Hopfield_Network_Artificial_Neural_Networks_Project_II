
import os
import numpy as np

from common.common_functions import check_accuracy, disturb_data, get_size_from_filename

class Test:
    def __init__(self,nn,train_data,name,disturb_coeff, iters, file_path):
        self.nn = nn
        self.train_data = train_data
        self.name = name
        self.disturb_coeff = disturb_coeff
        self.iters = iters
        self.file_path = file_path

    def start(self):
        np.random.seed(1)

        test_data = []
        disturb = int(self.disturb_coeff * self.nn.neuron_num)
        for pattern in self.train_data:
            test_data.append(disturb_data(pattern,disturb))

        train_data = np.asarray(self.train_data)
        test_data = np.asarray(test_data)

        self.nn.train_dataset(train_data)

        res = []
        for pattern in test_data:
            tmp = self.nn.predict(pattern, self.iters)
            res.append(check_accuracy(pattern,tmp))

        self.save_to_file(res)

    def cum_res(self, results):
        arr = np.array(results)
        return np.sum(arr)/len(arr)

    def save_to_file(self, results):
        path = os.path.join(self.file_path, self.name)
        f = open(path, "w")
        print("saving", self.name)
        f.write(f"Disturb test dataset: {self.disturb_coeff*100} %\n")
        f.write(f"Accuracy: {self.cum_res(results) * 100} %\n")
        f.write(f"Max iterations: {self.iters}\n")
        f.write(f"Data;Accuracy\n")
        results = [f'{str(i)};{str(results[i] * 100)}\n'
                for i in range(len(results))]
        f.writelines(results)
        f.close
