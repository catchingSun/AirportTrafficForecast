import numpy as np
import neurolab as nl
import pylab as pl


# input_data = pd.DataFrame(data={'Case' : [12, 23, 34, 24, 23, 2, 3, 4, 2, 4, 5, 7, 3, 12, 14, 18, 20]})
class FFNNModel:

    input_max = 0
    input_min = 0
    m = 8
    predict_count = 19

    def __init__(self):
        return

    # Normalized data
    def __normalized_data(self, input_data):
        input_data = input_data['passengerCount'].values
        input_data.dtype = 'float64'
        self.input_max = input_data.max()
        self.input_min = input_data.min()
        dis = self.input_max - self.input_min
        for i in range(len(input_data)):
            input_data[i] = (input_data[i] - self.input_min) / dis
        return input_data

    # Create train input and output data
    def __create_input_output_data(self, input_data):
        # input_data = self.__normalized_data(input_data)
        # print input_data
        n = len(input_data) - self.m
        input_array = np.zeros([n, self.m])
        output_array = np.zeros(n)
        for i in range(n):
            count = i
            for j in range(self.m):
                input_array[i, j] = input_data[count]
                count += 1
            output_array[i] = input_data[count]
        max_min = []
        row_max = input_array.max(axis=0)
        row_min = input_array.min(axis=0)
        for i in range(self.m):
            max_min.append([row_min[i], row_max[i]])
        return input_array, output_array, max_min

    # Create predict input data
    def __create_predict_input_data(self, input_data):
        n = len(input_data) - self.m + 1
        input_array = np.zeros([n, self.m])
        for i in range(n):
            count = i
            for j in range(self.m):
                input_array[i, j] = input_data[count]
                count += 1
        return input_array

    # Anti-normalize input data
    @staticmethod
    def __anti_normalization(input_data, data_max, data_min):
        for i in range(len(input_data)):
            input_data[i] = input_data[i] * (data_max - data_min) + data_min
        return input_data

    # Train BP neural networks
    def __train_ffnn(self, train_data):
        m = self.m
        res = self.__create_input_output_data(train_data)
        inp = res[0]
        tar = res[1]
        size = len(tar)
        tar = tar.reshape(size, 1)
        max_min = res[2]

        # Create network with 3 layers and random initialized
        net = nl.net.newff(max_min, [m, 1])
        # Train network
        error = net.train(inp, tar, epochs=500, show=100, goal=0.002)
        # Simulate network
        out = net.sim(inp)

        # pl.subplot(211)
        # pl.plot(error)
        # pl.xlabel("Epoch number")
        # pl.ylabel('error (default SSE)')
        #
        # pl.subplot(212)
        # # pl.plot(tar, '-', y3, '-')
        # pl.plot(tar, '-', out, '*')
        # pl.xlabel('Time / 10min')
        # pl.ylabel('Passenger Count')
        # pl.legend(['train target', 'net output'])
        # pl.show()
        return net

    # def __test_predict_data(self, true_data, predict_data):
    #     self.m
    #     return

    # Get prediction data
    def ffnn_model(self, input_data):
        normalized_data = self.__normalized_data(input_data)
        m = self.m
        test_data = normalized_data[-19:]
        train_data = normalized_data[:2111]

        net = self.__train_ffnn(train_data)
        predict_data = train_data[1:]

        for i in range(self.predict_count):
            predict_input_data = self.__create_predict_input_data(predict_data)
            temp = net.sim(predict_input_data)
            # print temp
            predict_data = np.delete(predict_data, 0)
            predict_data = np.append(predict_data, temp[-1])

        test_data = self.__anti_normalization(test_data, self.input_max, self.input_min)
        predict_data = predict_data[-19:]
        predict_data = self.__anti_normalization(predict_data, self.input_max, self.input_min)
        # pl.plot(test_data, '-', predict_data, '*')
        # pl.xlabel('Time / 10min')
        # pl.ylabel('Passenger Count')
        # pl.legend(['True data', 'predict output'])
        # pl.show()
        # result = predict_data[-19:]
        # print result
        return predict_data

# ffnn_model()
