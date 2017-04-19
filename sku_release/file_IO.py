# coding=utf-8

import pandas as pd
import os

class file_IO:
    def __init__(self):
        pass


    @staticmethod
    def read_xls(xls_name, sheet_idx=0, header=None):
        data = pd.read_excel(".\\data\\"+xls_name,sheetname=sheet_idx,header=header)
        # for c in range(data.shape[1]):
        #     data[c] = data[c].astype('unicode')

        # res = data.astype('unicode')
        return data

    @staticmethod
    def write_xls(var,xls_name,title=[]):
        if not os.path.exists('.\\output'):
            os.makedirs('.\\output')
        # writer = pd.ExcelWriter(".\\output\\" + xls_name)
        df = pd.DataFrame()
        if len(title)==0:
            df = pd.DataFrame(var)
        else:
            dc = {}
            for i,t in enumerate(title):
                dc[t] = var[i]

            df = pd.DataFrame(dc)
        df.to_excel(".\\output\\" + xls_name)


