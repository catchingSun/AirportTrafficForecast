import pandas as pd
import numpy as np


def read_file(raw_data_file):

    wifi_ap_file = "../RawData/Test_WIFI_AP_RAW_Data/"
    raw_data = pd.read_csv(raw_data_file)
    wifi_ap_name = raw_data[[0]].drop_duplicates().sort(columns='WIFIAPTag')
    wifi_ap_name.to_csv("ap_name.csv", index=False)
    for i in range(len(raw_data)):
        raw_data['timeStamp'][i] = raw_data['timeStamp'][i][:15] + '0'
    wifi_ap_grouped = raw_data.groupby(['WIFIAPTag'])
    #wifi_ap_grouped.aggregate(np.mean)
    for i in range(len(wifi_ap_name)):
        temp = wifi_ap_name.iloc[i]
        temp_group = wifi_ap_grouped.get_group(temp.values[0])
        temp_df = pd.DataFrame(data=temp_group.iloc[:, 1:3])
        temp_df = temp_df.sort(columns='timeStamp')
        temp_df.to_csv(wifi_ap_file+temp.values[0]+".csv", index=False)
    return
raw_file = "../RawData/WIFI_AP_Passenger_Records.csv"
read_file(raw_file)