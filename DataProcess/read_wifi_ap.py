import pandas as pd
import numpy as np


def spp(data):
    data = data[:2]
    return data


def read_file(raw_data_file):

    wifi_ap_file = "../RawData/WIFI_AP_RAW_Data/"
    raw_data = pd.read_csv(raw_data_file, nrows=5)
    wifi_ap_name = pd.DataFrame(data=raw_data['WIFIAPTag'])
    for i in range(len(raw_data)):
        raw_data['timeStamp'][i] = raw_data['timeStamp'][i][:15] + '0'
        wifi_ap_name['WIFIAPTag'][i] = wifi_ap_name['WIFIAPTag'][i][:2]
    wifi_ap_name = wifi_ap_name.drop_duplicates().sort(columns='WIFIAPTag')
    # print raw_data
    wifi_ap_grouped = raw_data.groupby(spp(raw_data['WIFIAPTag'].values))
    print wifi_ap_grouped.groups
'''    for j in wifi_ap_name['WIFIAPTag']:
        temp_df = pd.DataFrame()
        for i in raw_data['WIFIAPTag']:
            if j in i:
                # temp_df.append(raw_data[i])
                print raw_data[i]
                temp_df.to_csv(wifi_ap_file + j+".csv", index=False)
    return
'''

raw_file = "../RawData/WIFI_AP_Passenger_Records.csv"
read_file(raw_file)