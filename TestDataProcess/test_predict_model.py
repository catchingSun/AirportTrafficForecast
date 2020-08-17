# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import test_new_arima_model as nm


def arima_model(wifi_tag_passenger, predict_result_path):
    res_tup = nm._get_best_model(wifi_tag_passenger)
    results_ARIMA = res_tup[2]

    start = pd.to_datetime('2016-09-25 15:00', "%Y-%m-%d %H:%M")
    end = pd.to_datetime('2016-09-25 18:00', "%Y-%m-%d %H:%M")

    forecast_result = results_ARIMA.forecast(18)[0]
    forecast_result = list(forecast_result)
    # print results_ARIMA.summary()
    predict_result = results_ARIMA.predict(start=199, end=218)
    pd_predict_result = pd.DataFrame(predict_result, columns=['passengerCount']) \
        .between_time(start_time=start, end_time=end)
    pd_predict_result = pd.DataFrame(pd_predict_result)
    pd_predict_result.index.name = 'Time'
    # pd_predict_result['passengerCount'].values = forecast_result
    print pd_predict_result
    pd_predict_result.to_csv(predict_result_path)

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

