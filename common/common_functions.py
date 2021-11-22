import numpy as np
import pandas as pd

def read_data_as_vectors(path):
    try:
        return pd.read_csv(path).values.tolist()
    except:
        raise FileNotFoundError('Could not find the file')

def read_data_as_arrays(path, size):
    result = []
    for d in read_data_as_vectors(path):
        result.append(data_to_array(d, (size)))
    return result

def data_to_array(data, size):
    return np.array(data).reshape(size)

def array_to_vector(array):
    return np.reshape(np.array(array), -1)

def disturb_data(vec, num=10):
    vec_d= vec[:]
    selected = np.random.choice(len(vec), num, replace=False)
    for idx in selected:
        vec_d[idx] = -vec_d[idx]
    return vec_d
    
    