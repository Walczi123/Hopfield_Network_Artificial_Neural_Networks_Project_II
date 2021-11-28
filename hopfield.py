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
        # self.train(train_data)

        # for i in range(iter):
        #     for x in train_data:
        #         y = np.matmul(self.weights, x)
        #         for r in range(self.neuron_num):
        #             for c in range(self.neuron_num):
        #                 self.weights[r,c] += n * y[r] * (x[c] - y[r] * self.weights[r][c])

        # self.train(train_data)

        for i in range(iter):
            Wprev = self.weights.copy()
            print(f"it: {i}   NORM: {np.linalg.norm(self.weights,2)}")
            for x in train_data:
                y = np.matmul(self.weights, x)
                # y = np.sign(y)
                wd = np.matmul(self.weights,y)
                xd = x - wd
                d2 = np.outer(n*y,xd)
                self.weights += d2
                # for i in range(self.neuron_num):
                #     for j in range(self.neuron_num):
                #         y2 = y[i]*y[i]
                #         wy = self.weights[i,j]*y[i]
                #         xwy = x[j] - self.weights[i,j]*y[i]
                #         self.weights[i,j] += n*y[i]*(x[j] - self.weights[i,j]*y[i])


            if np.linalg.norm(Wprev - self.weights) < 1e-14:
                break 
        
        self.weights -= np.diag(np.diag(self.weights))


    def train_oja2(self, train_data, iter, n):
        for _ in range(iter):
            y = self.weights * train_data
            # y = np.sign(y)
            wd = self.weights * y
            xd = train_data - wd
            d2 = (n*y) * xd
            self.weights += d2

        self.weights -= np.diag(np.diag(self.weights))