
import numpy as np
from features.preprocessing.helpers import take
import time


arr = np.random.randint(0, high = 10, size = (1000,2000,8))

start_time = time.time()

for i in range(200):
    x1 = take(arr, np.arange(i+5, i+1015), axis = 1)

print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()

for i in range(200):
    x2 = np.take(arr, np.arange(i+5, i+1015), axis = 1)

print("--- %s seconds ---" % (time.time() - start_time))

print(f'max difference: {np.abs(x1 - x2).max()}')