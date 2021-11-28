from tkinter.constants import NO
import numpy as np
import pandas as pd
import cv2

def binary(x):
    if x<= 0:
        return -1
    return 1

def read_data_as_vectors(path):
    try:
        return pd.read_csv(path, header=None).values.tolist()
    except:
        raise FileNotFoundError('Could not find the file')

def read_image(path, mode = cv2.IMREAD_GRAYSCALE):
    try:
        return cv2.imread(path, mode);
    except:
        raise FileNotFoundError('Could not find the file')

def read_data_as_arrays(path, size):
    result = []
    for d in read_data_as_vectors(path):
        result.append(data_to_array(d, (size)))
    return result

def resize_image(image, size):
    return cv2.resize(image, size)

def read_image_as_arrays(path, threshold = 123, resize = None):
    image = read_image(path)
    if resize is not None:
        image = resize_image(image, resize)
    array = np.array(image)
    if threshold is not None:
        th = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
        vbinary = np.vectorize(binary)
        array = vbinary(np.array(th))
    return array

def read_image_as_vector(path, threshold = 123, resize = None):
    return array_to_vector(read_image_as_arrays(path, threshold, resize))

def data_to_array(data, size):
    return np.array(data).reshape(size)

def array_to_vector(array):
    return np.reshape(np.array(array), -1)

def disturb_data(vec, num=10):
    vec_d= vector_deep_copy(vec)
    selected = np.random.choice(len(vec), num, replace=False)
    for idx in selected:
        vec_d[idx] = -vec_d[idx]
    return vec_d

def vector_deep_copy(vec):
    return [x for x in vec]
    
def check_accuracy(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0
    vec1 = vector1 + vector2
    vec = np.where(vec1 != 0)[0]
    return len(vec)/len(vector1)

    # s = 0
    # if len(vector1) != len(vector2):
    #     return s
    # len_vec = len(vector1)
    # for x in range(len_vec):
    #     if vector1[x] == vector2[x]:
    #         s+=1
    # return s/len_vec

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def get_size_from_filename(filename):
    s = filename.split('.')[0].split('-')
    for p in s:
        if has_numbers(p):
            res = p.split('x')
            return (int(res[0]), int(res[1]))
    raise "invalid file"