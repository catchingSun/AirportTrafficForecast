import pandas as pd
import numpy as np


def read_file():

    df = pd.DataFrame(data={'Case' : ['Ac67','Ae','As','Bs','Ad','Af','Bv','As','Af'],
                            'Data' : np.random.randn(9)})
    df_name = pd.DataFrame(data=df['Case'])
    # print df_name
    for i in range(len(df_name)):
        df_name['Case'][i] = df_name['Case'][i][0]
    #
    df_name = df_name.drop_duplicates().sort(columns='Case')
    # print df_name
    for i in df['Case']:
        for j in df_name['Case']:
            if j in i:
                print j
read_file()