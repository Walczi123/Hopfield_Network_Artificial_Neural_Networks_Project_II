import numpy as np

class HopfieldNetwork():     
    def __init__(self, neuron_num, num_iter):
        self.neuron_num = neuron_num
        self.weights = np.empty(self.neuron_num)
        self.num_iter = num_iter

    def train(self, train_data):
        for data in train_data:
            w = np.outer(data, data)
            diag_w = np.diag(np.diag(w))
            w = w - diag_w
            w = w / w.shape[0]
            self.weights = self.weights + w

    def predict(self, data):
        # return np.sign(self.weights @ data)
        return self.sync_predict(data)

    def energy(self, state):
        return  -0.5*np.matmul(np.matmul(state, self.weights), state)

    def sync_predict(self, data):
        tmp = data
        e = self.energy(tmp)

        for i in range(self.num_iter):
            tmp = np.sign(self.weights @ tmp)
            e_new = self.energy(tmp)
            if e == e_new:
                return tmp
            e = e_new
        return tmp

    def async_predict(self, data):
        pass