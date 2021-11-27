import numpy as np
from common.common_functions import vector_deep_copy

class HopfieldNetwork():     
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num
        self.weights = np.zeros([self.neuron_num, self.neuron_num])

    def train(self, train_data):
        w = np.outer(train_data, train_data)
        diag_w = np.diag(np.diag(w))
        w = w - diag_w
        w = w / w.shape[0]
        self.weights = self.weights + w

    def train_dataset(self, train_datas):
        for data in train_datas:
            self.train(data)

    def predict(self, data, num_iter, use_async = False, async_iter = 100):
        if not use_async:
            return self.sync_predict(data, num_iter)
        else:
            return self.async_predict(data, num_iter, async_iter)

    def energy(self, state):
        return  -0.5*np.matmul(np.matmul(state, self.weights), state)

    def sync_predict(self, data, num_iter):
        tmp = vector_deep_copy(data)
        e = self.energy(tmp)

        for i in range(num_iter):
            tmp = np.sign(self.weights @ tmp)
            e_new = self.energy(tmp)
            if e == e_new:
                return tmp
            e = e_new
        return tmp

    def async_predict(self, data, num_iter, async_iter):
        tmp = vector_deep_copy(data)
        e = self.energy(tmp)
        for i in range(async_iter):
            for j in range(100):
                idx = np.random.randint(0, self.neuron_num) 
                tmp[idx] = np.sign(self.weights[idx].T @ tmp)
                        
            e_new = self.energy(tmp)
            
            if e == e_new:
                return tmp
            e = e_new
        return tmp