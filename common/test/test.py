
import os
import numpy as np

from common.common_functions import check_accuracy, disturb_data, get_size_from_filename

class Test:
    def __init__(self,nn,train_data,name,disturb_coeff, oja_learning_rate, iters, async_iters, oja_iters, file_path):
        self.nn = nn
        self.train_data = train_data
        self.name = name
        self.disturb_coeff = disturb_coeff
        self.iters = iters
        self.async_iters = async_iters
        self.file_path = file_path
        self.oja_learning_rate = oja_learning_rate
        self.oja_iters = oja_iters

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

    def start_all(self):
        np.random.seed(1)

        test_data = []
        disturb = int(self.disturb_coeff * self.nn.neuron_num)
        for pattern in self.train_data:
            test_data.append(disturb_data(pattern,disturb))

        train_data = np.asarray(self.train_data)
        test_data = np.asarray(test_data)

        self.nn.train_dataset(train_data)

        res = []
        cycles = []
        for pattern in test_data:
            tmp = self.nn.predict(pattern, self.iters, check_cycle = True)
            res.append(check_accuracy(pattern,tmp[0]))
            cycles.append(tmp[1])

        res_async = []
        for pattern in test_data:
            tmp = self.nn.predict(pattern, self.iters, True, self.async_iters)
            res_async.append(check_accuracy(pattern,tmp))
        
        self.nn.train_oja(train_data, self.oja_iters, self.oja_learning_rate)

        res_oja = []
        cycles_oja = []
        for pattern in test_data:
            tmp = self.nn.predict(pattern, self.iters, check_cycle = True)
            res_oja.append(check_accuracy(pattern,tmp[0]))
            cycles_oja.append(tmp[1])

        res_async_oja = []
        for pattern in test_data:
            tmp = self.nn.predict(pattern, self.iters, True, self.async_iters)
            res_async_oja.append(check_accuracy(pattern,tmp))

        self.save_to_file_all(res, res_async, res_oja, res_async_oja, cycles, cycles_oja)

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

    def save_to_file_all(self, results, results_async, results_oja, results_oja_async, cycles, cycles_oja):
        path = os.path.join(self.file_path, self.name + f"_{str(int(self.disturb_coeff * 100))}")
        res = [results, results_async, results_oja, results_oja_async]
        f = open(path, "w")
        print("saving", self.name)
        f.write(f"Disturb test dataset: {self.disturb_coeff*100} %\n")
        f.write(f"Max iterations: {self.iters}\n")
        f.write(f"Max async iterations: {self.async_iters}\n")
        f.write(f"Max oja iterations: {self.oja_iters}\n")
        f.write(f"Oja learning rate: {self.oja_learning_rate}\n")
        f.write(f"\t\t\t\t\tRES\t\t\tRES_ASYNC\t\t\tRES_OJA\t\t\tRES_OJA_ASYNC\n")
        f.write(f"Accuracy:\t\t {round(self.cum_res(results) * 100,2)} %\t\t\t{round(self.cum_res(results_async) * 100,2)} %\t\t\t{round(self.cum_res(results_oja) * 100,2)} %\t\t\t{round(self.cum_res(results_oja_async) * 100,2)} %\n")
        for i in range(max(len(results), len(results_async), len(results_oja), len(results_oja_async))):
            tmp = ""
            for j in range(len(res)):
                if len(res[j]) > i:
                    tmp += f"\t\t\t{str(round(res[j][i] * 100, 2))}"
                    if j == 0:
                        if cycles[j]:
                            tmp += f"\tC"
                    if j == 2:
                        if cycles_oja[j]:
                            tmp += f"\tC"
            f.write(f"{str(i)}:\t\t{tmp}\n")
        f.close
