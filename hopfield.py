import numpy as np

class HopfieldNetwork():     
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num
        self.weights = np.empty(10)

    def train(self, train_data):
        for data in train_data:
            w = np.outer(data, data)
            diag_w = np.diag(np.diag(w))
            w = w - diag_w
            w = w / w.shape[0]
            self.weights = self.weights + w

    def predict(self, data):
        return self.sync_predict(data)

    def energy(self, state):
        return

    def sync_predict(self, data):
        # Compute initial state energy
        state = data
        e = self.energy(state)
        
        # Iteration
        for i in range(self.num_iter):
            # Update s
            tmp = np.sign(self.W @ tmp - self.threshold)
            # Compute new state energy
            e_new = self.energy(tmp)
            
            # s is converged
            if e == e_new:
                return tmp
            # Update energy
            e = e_new
        return tmp

    def async_predict(self, data):
        pass