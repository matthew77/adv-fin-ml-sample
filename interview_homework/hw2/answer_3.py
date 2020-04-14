import pandas as pd
import numpy as np
import os

# DATA_HOME = "E:\\tmp\\data\\interview\\hw2"
DATA_HOME = "./.."
# load trading date from answer 1
fp_trading_date = os.path.join(DATA_HOME, "answer", "answer1_tradingdate.csv")
trading_date_df = pd.read_csv(fp_trading_date, 
                    parse_dates=["date"],
                    dtype={'date_str': str}).set_index(['date'])
# load data from dpx
df_rt = pd.DataFrame()
for d in trading_date_df['date_str']:
    fp_dpx = os.path.join(DATA_HOME, 'dpx', f"{d}.dpx.csv")
    tmp_df = pd.read_csv( fp_dpx,
        usecols=['TradingDay','UNIQUE', 'CLOSE'],
        parse_dates=["TradingDay"],
    )
    df_rt = df_rt.append(tmp_df.copy())

# df_rt['daily_rt'] = df_rt['CLOSE']/df_rt['PrevClosePrice'] - 1
# df_rt.drop(columns=['CLOSE', 'PrevClosePrice'], inplace=True)
df_rt = df_rt.set_index(['TradingDay', 'UNIQUE'])
df_rt = df_rt.unstack(level=-1)
# calculate rolling mean
df_rolling_mean = df_rt.rolling(10).mean()
# remove the 1st level column
multi_col_idx = df_rolling_mean.columns
single_col_idx = pd.Index([e[1] for e in multi_col_idx.tolist()])
df_rolling_mean.columns = single_col_idx
# export results
df_export = os.path.join(DATA_HOME, "answer", "answer3_rolling_mean.csv")
df_rolling_mean.to_csv(df_export)