# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import test_new_arima_model as nm


def arima_model(wifi_tag_passenger, predict_result_path):

    res_tup = nm._get_best_model(wifi_tag_passenger)
    results_ARIMA = res_tup[2]
    start = pd.to_datetime('2016-09-25 15:00', "%Y-%m-%d %H:%M")
    end = pd.to_datetime('2016-09-25 18:00', "%Y-%m-%d %H:%M")
    forecast_count = 19
    forecast_result = results_ARIMA.forecast(forecast_count)[0]
    times = pd.date_range(start=start, end=end, freq='10Min')
    forecast_result = pd.DataFrame(forecast_result, columns=['passengerCount'], index=times)
    forecast_result.index.name = 'Time'
    forecast_result.to_csv(predict_result_path)

    return


def predict_model(wifi_tag_file, ap_name):

    wifi_tag =pd.read_csv(wifi_tag_file)
    for i in range(len(wifi_tag)):
        wifi_tag['timeStamp'][i] = wifi_tag['timeStamp'][i][:15]+'0'
    wifi_tag_grouped = wifi_tag.tail(2000).groupby('timeStamp')
    wifi_tag_passenger = wifi_tag_grouped.aggregate(np.mean)
    wifi_tag_passenger.index = pd.to_datetime(wifi_tag_passenger.index, format='%Y-%m-%d-%H-%M')
    wifi_tag_passenger = pd.DataFrame.sort_index(wifi_tag_passenger)
    path = "../predict_result/"
    ap_name = path + ap_name
    arima_model(wifi_tag_passenger, ap_name)

'''
def printft(wifi_tag_file, ap_name):
    print wifi_tag_file
    wifi_tag =pd.read_csv(wifi_tag_file)
    print wifi_tag
'''
predict_model("../RawData/WIFI_AP_RAW_Data/E1-1A-4<E1-1-04>.csv", "E1-1A-4<E1-1-04>")

