import matplotlib.pyplot as plt
import numpy as np

def m_average(lis):
    result = lis.copy()
    for i in range(len(lis)):
        result[i] = np.mean(result[:i+1])
    return result