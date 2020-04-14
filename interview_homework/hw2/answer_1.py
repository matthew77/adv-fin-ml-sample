import pandas as pd
import numpy as np
import os
import codecs

# read tradingDate.csv
# DATA_HOME = "E:\\tmp\\data\\interview\\hw2"
DATA_HOME = "./.."

date_list = list()
fp = os.path.join(DATA_HOME, "tradingDate", "tradingDate.csv")

trading_dates = pd.read_csv(fp, header=None, names=['date_str'])
# remove the item start with '#'
trading_dates = trading_dates[trading_dates['date_str'].str.startswith('#')==False]
date_idx = pd.to_datetime(trading_dates['date_str'])
trading_dates = trading_dates.set_index(date_idx)
trading_dates.index.rename('date', inplace=True)
# select the range start from 2013.01.01
# 从dpx文件夹可知最近的日期为 2014-01-03
trading_dates_start_from_2013 = trading_dates['2013-01-01':'2014-01-03']
fp_output = os.path.join(DATA_HOME, "answer", "answer1.txt")
with codecs.open(fp_output, 'a', encoding='utf8') as f:
    print("2013.01.01至今之间的交易日的数量: {0}".format(trading_dates_start_from_2013.shape[0]), file=f)
    print("具体交易日为：", file=f)
    for d in trading_dates_start_from_2013.index:
        print(trading_dates_start_from_2013.loc[d]['date_str'], file=f)
# export the result set
df_export = os.path.join(DATA_HOME, "answer", "answer1_tradingdate.csv")
trading_dates_start_from_2013.to_csv(df_export)
    