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

    def predict(self, data, num_iter, use_async = False, async_iter = 100, check_cycle = False):
        if not use_async:
            if not check_cycle:
                return self.sync_predict(data, num_iter)[0]
            else:
                return self.sync_predict(data, num_iter)
        else:
            return self.async_predict(data, num_iter, async_iter)

    def energy(self, state):
        return  -0.5*np.matmul(np.matmul(state, self.weights), state)

    def sync_predict(self, data, num_iter):
        tmp = vector_deep_copy(data)
        e = self.energy(tmp)
        history = [tmp, tmp]
        cycle = False
        for i in range(num_iter):
            tmp = np.sign(self.weights @ tmp)
            if list(tmp) == list(history[0]) and i > 0:
                cycle = True
            else:
                history[0] = history[1]
                history[1] = tmp
            e_new = self.energy(tmp)
            if e == e_new:
                return tmp, cycle
            e = e_new
        return tmp, cycle

    def async_predict(self, data, num_iter, async_iter):
        tmp = vector_deep_copy(data)
        e = self.energy(tmp)
        for i in range(num_iter):
            for j in range(async_iter):
                idx = np.random.randint(0, self.neuron_num) 
                tmp[idx] = np.sign(self.weights[idx].T @ tmp)
                        
            e_new = self.energy(tmp)
            
            if e == e_new:
                return tmp
            e = e_new
        return tmp

    def train_oja(self, train_data, iter, n):
        for _ in range(iter):
            Wprev = self.weights.copy()
            for x in train_data:
                y = np.matmul(self.weights, x)
                self.weights += np.outer(n*y,x - np.matmul(self.weights,y))

            if np.linalg.norm(Wprev - self.weights) < 1e-14:
                break 
        
        self.weights -= np.diag(np.diag(self.weights))


    def train_oja2(self, train_data, iter, n): 
        for _ in range(iter):
            Wprev = self.weights.copy()
            y = np.matmul(self.weights, train_data)
            wd = np.matmul(self.weights,y)
            xd = train_data - wd
            d2 = np.outer(n*y,xd)
            self.weights += d2

            if np.linalg.norm(Wprev - self.weights) < 1e-14:
                break 

        self.weights -= np.diag(np.diag(self.weights))