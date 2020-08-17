# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import new_arima_model as nm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox


def test_stationarity(timeseries):
    dftest = ADF(timeseries['passengerCount'], autolag='AIC')
    # print(u'原序列ADF检验值:')
    # print dftest
    # plot_pacf(timeseries).show()
    # plot_acf(timeseries['passengerCount']).show()


def test_noise(timeseries):
    noise = acorr_ljungbox(timeseries['passengerCount'], lags=1)
    # print u'原序列的白噪声检验结果为：'
    # print 'stat                  | p-value'
    # for x in noise:
    #     print x,'|',


def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rolmean = pd.rolling_mean(timeSeries['passengerCount'], window=size)
    rolstd = pd.rolling_std(timeSeries['passengerCount'], window=size)
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries['passengerCount'], span=size)
    #orig = plt.plot(timeSeries['passengerCount'])
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    # timeSeries.plot(color='blue', label='Original')
    # rol_weighted_mean.plot(color='green', label='Rolling Weighted Mean')
    # rolmean.plot(color='red', label='Rolling Mean')
    # rolstd.plot(color='black', label='Rolling Std')
    #
    #
    #
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.legend(loc='best')
    # plt.show()


def arima_model(wifi_tag_passenger):
    # test_stationarity(wifi_tag_passenger)
    # test_noise(wifi_tag_passenger)
    # draw_trend(wifi_tag_passenger, 1)
    res_tup = nm._get_best_model(wifi_tag_passenger)
    order = res_tup[1]
    results_ARIMA = res_tup[2]
    # results_ARIMA = sm.tsa.ARIMA(wifi_tag_passenger, order=(6,1,0)).fit(
    #     method='css', trend='nc', disp=False
    # )
    # print order
    start = pd.to_datetime('2016-09-25 15:00', "%Y-%m-%d %H:%M")
    end = pd.to_datetime('2016-09-25 18:00', "%Y-%m-%d %H:%M")

    print results_ARIMA.forecast(30)[0]
    # fig = plt.plot(fig, color='green')
    # ax, f1 = plt.subplots(figsize=(10,8))
    ax=results_ARIMA.predict()
    ax = plt.plot(ax, color='red')

    # f1 = plt.plot(wifi_tag_passenger, color='blue')

    fig, ax = plt.subplots(figsize=(10,8))
    ax = wifi_tag_passenger.plot(color='red')
    # fig = results_ARIMA.plot_predict(start, end, ax=ax, dynamic=True, plot_insample=False)
    plt.xlabel("time")
    plt.ylabel("passagerCounts")
    plt.show()


def arima_model(wifi_tag_file):

    #dateparse = lambda dates: pd.datetime.strptime(wifi_tag, '%Y-%m-%d-%H-%M')
    #df =pd.read_csv(wifi_tag_file, parse_dates=['timeStamp'], index_col='timeStamp', date_parser=dateparse)

    wifi_tag =pd.read_csv(wifi_tag_file)
    for i in range(len(wifi_tag)):
        wifi_tag['timeStamp'][i] = wifi_tag['timeStamp'][i][:15]+'0'
    wifi_tag_grouped = wifi_tag.tail(2000).groupby('timeStamp')
    wifi_tag_passenger = wifi_tag_grouped.aggregate(np.mean)
    wifi_tag_passenger.index = pd.to_datetime(wifi_tag_passenger.index, format='%Y-%m-%d-%H-%M')
    wifi_tag_passenger = pd.DataFrame.sort_index(wifi_tag_passenger)
    arima_model(wifi_tag_passenger)
    # print wifi_tag_passenger.index
    #
    # print results_ARIMA.forecast(100)[0]
    # print results_ARIMA.summary()

    # temp = results_ARIMA.forecast()[0]
    # predictions.append(temp)
    # print predictions
'''    wifi_tag_passenger.plot(label='Original')
    (results_ARIMA.fittedvalues).plot(color='red', label='Prediction')
    plt.legend(loc='best')
    plt.show()
'''

    #plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-wifi_tag_passenger['passengerCount'])**2))
    #
    # #给出一份模型报告
    # results_ARIMA.summary2()
    #
    # #作为期5天的预测，返回预测结果、标准误差、置信区间。


#
arima_model("../RawData/WIFI_AP_RAW_Data/E1-1A-2<E1-1-02>.csv")

