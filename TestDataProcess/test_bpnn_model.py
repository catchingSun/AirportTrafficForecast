import numpy as np
import pandas as pd
import neurolab as nl
import pylab as pl


def process_data():
    input_data = pd.DataFrame(data={'Case' : [12, 23, 34, 24, 23, 2, 3, 4, 2, 4, 5, 7, 3, 12, 14, 18, 20]})
    input_data = input_data['Case'].values

    input_data.dtype = 'float64'
    input_max = input_data.max()
    input_min = input_data.min()
    dis = input_max - input_min
    for i in range(len(input_data)):
        input_data[i] = (input_data[i] - input_min) / dis
    # print input_data
    m = 8
    n = len(input_data) - m
    input_array = np.zeros([n, m])
    output_array = np.zeros(n)
    for i in range(n):
        count = i
        for j in range(m):
            input_array[i, j] = input_data[count]
            count += 1
        output_array[i] = input_data[count]
    max_min = []
    row_max = input_array.max(axis=0)
    row_min = input_array.min(axis=0)
    for i in range(m):
        max_min.append([row_min[i], row_max[i]])
    return input_array, output_array, max_min


def bpnn_model():

    res = process_data()
    input_data = res[0]

    tar = res[1]

    size = len(tar)
    tar = tar.reshape(size, 1)
    max_min = res[2]
    # Create network with 2 layers and random initialized
    net = nl.net.newff(max_min,[8, 1])
    # Train network
    error = net.train(input_data, tar, epochs=500, show=100, goal=0.000001)

    # Simulate network
    out = net.sim(input_data)
    print 'out : '
    print out
    pl.subplot(211)
    pl.plot(error)
    pl.xlabel('Epoch number')
    pl.ylabel('error (default SSE)')

    x2 = np.linspace(-6.0,6.0,150)
    # y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

    y3 = out.reshape(size)

    pl.subplot(212)
    pl.plot(tar, '*', y3, '-')
    # pl.legend(['train target', 'net output'])
    pl.show()
    return

bpnn_model()