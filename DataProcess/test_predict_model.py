# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import new_arima_model as nm


def arima_model(wifi_tag_passenger):

    res_tup = nm._get_best_model(wifi_tag_passenger)
    results_ARIMA = res_tup[2]

    start = pd.to_datetime('2016-09-25 15:00', "%Y-%m-%d %H:%M")
    end = pd.to_datetime('2016-09-25 18:00', "%Y-%m-%d %H:%M")

    print results_ARIMA.forecast(30)[0]


def arima_model(wifi_tag_file):

    wifi_tag =pd.read_csv(wifi_tag_file)
    for i in range(len(wifi_tag)):
        wifi_tag['timeStamp'][i] = wifi_tag['timeStamp'][i][:15]+'0'
    wifi_tag_grouped = wifi_tag.tail(2000).groupby('timeStamp')
    wifi_tag_passenger = wifi_tag_grouped.aggregate(np.mean)
    wifi_tag_passenger.index = pd.to_datetime(wifi_tag_passenger.index, format='%Y-%m-%d-%H-%M')
    wifi_tag_passenger = pd.DataFrame.sort_index(wifi_tag_passenger)
    arima_model(wifi_tag_passenger)


if __name__ == '__main__':
    path = "../RawData/WIFI_AP_RAW_Data/E1-1A-3<E1-1-03>.csv"
    arima_model(path)

