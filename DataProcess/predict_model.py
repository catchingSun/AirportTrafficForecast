# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import new_arima_model as nm
# import matplotlib.pyplot as plt
import pylab as pl
from ffnn_model import FFNNModel
from sklearn import metrics


def model_test(results_ARIMA):
    resid = results_ARIMA.resid
    r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1,41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    table.set_index('lag')
    if (table['Prob(>Q)'][5] > 0.05) and (table['Prob(>Q)'][11] > 0.05):
        return 1
    return 0


def get_forecast_result(results_ARIMA):

    start = pd.to_datetime('2016-09-25 11:50', "%Y-%m-%d %H:%M")
    end = pd.to_datetime('2016-09-25 14:50', "%Y-%m-%d %H:%M")
    forecast_count = 19
    forecast_result = results_ARIMA.forecast(forecast_count)[0]
    times = pd.date_range(start=start, end=end, freq='10Min')
    forecast_result = pd.DataFrame(forecast_result, columns=['passengerCount'], index=times)
    # forecast_result.index.name = 'Time'
    # forecast_result.to_csv(predict_result_path)
    return forecast_result


def calculate_mean_square_error(ap_name, true_data, forecast_data, model_type):
    # mse = ((forecast_data - true_data) ** 2).mean()
    mse = metrics.mean_squared_error(true_data, forecast_data)
    # print true_data
    # print forecast_data
    # print model_type, 'mse : '
    # print mse
    data = [ap_name, model_type, mse]
    path = "../predict_result/error/mean_square_error.csv"
    mse_file = open(path, 'a+')
    mse_file.write(str(data))
    mse_file.write("\n")
    mse_file.close()


def arima_model(wifi_tag_passenger, true_data, ap_name):

    predict_data = wifi_tag_passenger.head(200)

    res_tup = nm._get_best_model(predict_data)

    results_ARIMA = res_tup[2]
    # print res_tup[1]
    model_test_result = model_test(results_ARIMA)

    if model_test_result == 1:
        forecast_data = get_forecast_result(results_ARIMA)
        calculate_mean_square_error(ap_name, true_data['passengerCount'], forecast_data['passengerCount'], 'arima')
    # pl.plot(true_data, '-', forecast_data, 'x')
    # pl.xlabel('Time / 10min')
    # pl.ylabel('Passenger Count')

    forecast_error = forecast_data - true_data
    temp = pd.merge(true_data, forecast_data, left_index=True, right_index=True)

    forecast_data = pd.merge(temp, forecast_error, left_index=True, right_index=True)
    forecast_data.columns = ['true_data', 'arima_forecast_result', 'arima_forecast_error']
    # print forecast_data


    return forecast_data


def ffnn_model(ap_name, wifi_tag_passenger, true_data):
    bm = FFNNModel()
    ffnn_forecast = bm.ffnn_model(wifi_tag_passenger)
    ffnn_forecast = pd.DataFrame(ffnn_forecast, columns=['passengerCount'])
    ffnn_forecast.index = true_data.index
    calculate_mean_square_error(ap_name, true_data, ffnn_forecast, 'ffnn')
    ffnn_forecast_error = ffnn_forecast - true_data
    ffnn_forecast.columns = ['ffnn_forecast_result']
    ffnn_forecast_error.columns = ['ffnn_forecast_error']
    ffnn_forecast_result = pd.merge(ffnn_forecast, ffnn_forecast_error, left_index=True, right_index=True)

    return ffnn_forecast_result


def predict_model(wifi_tag_file, ap_name):

    wifi_tag =pd.read_csv(wifi_tag_file)
    for i in range(len(wifi_tag)):
        wifi_tag['timeStamp'][i] += '0'
    wifi_tag_grouped = wifi_tag.groupby('timeStamp')
    wifi_tag_passenger = wifi_tag_grouped.aggregate(np.mean)
    wifi_tag_passenger.index = pd.to_datetime(wifi_tag_passenger.index, format='%Y-%m-%d-%H-%M')
    # print wifi_tag_passenger
    wifi_tag_passenger.index.name = 'Time'
    wifi_tag_passenger = pd.DataFrame.sort_index(wifi_tag_passenger)
    # wifi_tag_passenger.plot()
    # plt.show()
    true_data = wifi_tag_passenger.tail(19)

    path = "../predict_result/value/"
    predict_result_path = path + ap_name
    arima_forecast = arima_model(wifi_tag_passenger.tail(219), true_data, ap_name)
    ffnn_forecast = ffnn_model(ap_name, wifi_tag_passenger, true_data)
    forecast = pd.merge(arima_forecast, ffnn_forecast, left_index=True, right_index=True)
    forecast.to_csv(predict_result_path)
    # pl.plot(true_data, '-', forecast['arima_forecast_result'], '*')
    # pl.xlabel('Time / 10min')
    # pl.ylabel('Passenger Count')
    # pl.legend(['True data', 'ARIMA output'])
    # pl.show()
    # pl.plot(true_data, '-', forecast['ffnn_forecast_result'], '*')
    # pl.xlabel('Time / 10min')
    # pl.ylabel('Passenger Count')
    # pl.legend(['True data', 'FFNN output'])
    # pl.show()
    return

'''
def printft(wifi_tag_file, ap_name):
    print wifi_tag_file
    wifi_tag =pd.read_csv(wifi_tag_file)
    print wifi_tag
'''
# predict_model("../RawData/WIFI_AP_RAW_Data/E1-1A-1<E1-1-01>.csv", "E1-1A-1<E1-1-01>")

