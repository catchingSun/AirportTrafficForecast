import multiprocessing as mp
import pandas as pd
import predict_model as model


def process_function(wifi_ap_name_list):
    # time.sleep(2)

    path = "../RawData/WIFI_AP_RAW_Data/"
    for i in wifi_ap_name_list:
        # print i
        ap_path = path + i + ".csv"
        model.predict_model(ap_path, i)
        # model.printft(ap_path, i)
    return


def get_wifi_ap_name():
    wifi_ap_name = pd.read_csv("../RawData/WIFI_AP_RAW_Data/ap_name.csv")
    wifi_ap_name_list = wifi_ap_name['WIFIAPTag'].values
    return wifi_ap_name_list


def apply_async_with_callback(wifi_ap_name_list, process_count):
    pool = mp.Pool(processes=process_count)
    temp = 0
    for i in xrange(process_count):
        pool.apply_async(process_function, args=(wifi_ap_name_list[i], ))
    pool.close()
    pool.join()

if __name__ == '__main__':
    wifi_ap_name_list = get_wifi_ap_name()
    # wifi_ap_name_list = wifi_ap_name_list[0:4]
    wifi_ap_count = len(wifi_ap_name_list)
    # print wifi_ap_count
    process_count = 4
    split_segment = wifi_ap_count / process_count
    wifi_ap = list()
    temp = 0
    for i in range(process_count):

        if i == process_count - 1:
            temp = temp
            wifi_ap.append(wifi_ap_name_list[temp:])
        else:
            temp = (i + 1) * split_segment
            wifi_ap.append(wifi_ap_name_list[i * split_segment:temp])
    apply_async_with_callback(wifi_ap, process_count)