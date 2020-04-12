"""
问题：

2.如果某个分类特征由 N 个数值特征组合，应使用分类特征还是涵盖
更多信息的全部 N 个数值特征（N 较小） ？ 例如： 情况 1： 使用 A, B,
C 三个因子； 情况 2： 使用 A*(B>C) 因子（无需代码， 只需说明）

答：我倾向于两种情况同时使用。合成一个新的特征，如果新合成的特征对分类有更强的解释性，例如，房子的特征
包含了L（长），W（宽）， 但是面积（S=W*L）可能是一个更好的特征，可以将L,W,S 这3个特征都加入模型进行训练。
或者，在训练之前，进行特征选择，保留有意义的特征。
"""

"""
## 数据分析/清理

数据特征：
- 不均衡，TargetFeature = 1 只占1%左右
- 数据本身是time series，因此具有明显的auto correlation

因此需要对以上2种情况做处理。

"""

"""
###数据采样

由于time series 数据具有auto correlation的特性，那么，相邻的数据具有很高的相关性。因此，如果直接使用这些数据，那么这些数据
很明显就不是IID，这样训练出来的模型效果会受到影响。

这里使用一种简单的方法，就是随机间隔采样，用这种方式对负样本进行采样，但是完整保留正样本。因为，正样本实在太少了，因此受auto correlation
的影响可以忽略。

强调：这里的目的不是欠采样！！！
"""

import pandas as pd
import numpy as np
import os
import talib
from random import randrange
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

DATA_HOME = "E:\\tmp\\data\\interview"
LABELS = ("Stock_1", "Stock_2", "Stock_3", "Stock_4", "Stock_5", "Stock_6", "Stock_7", 
                "Stock_8", "Stock_9", "Stock_10")
LOOK_BACK_PERIOD = [1,3,5,8,13,21,34,55]  # 回看时间长度
SAMPLE_INTERVAL = [3, 15]  # 数据采样间隔
ROLLING_WINDOW = 60    #计算滑动平均窗口长度


origin_data = dict()
 

    '''
    回溯周期=15
    
    price feature 里面有关直接的价格不用。price feature包括

    - log return + look back
    - momentum + 累计
    - BBbands 是否超上界 + look back
                是否超下界限 + look back
    - EMA， 当前价格到EMA的% + look back （optional）
    - aroon, + look back （optional，可以考虑以前到今天的指标变化的数据）
    ----------------------------
    volume feature：
    - log return + look back
    - OBV -- 不能直接用，需要以return的方式look back
    ---------------------------
    tape feature:

    --------------------------
    order feature:
    做成 rolling的 z值， + look back
    --------------------------
    '''

    # ----------------------- clean data and create features -------------------
def preprocess_data(label):
    file_path = os.path.join(DATA_HOME, f'{label}.csv')
    raw_data = pd.read_csv(file_path, index_col=0)
    # 处理NaN单元格
    raw_data.fillna(method='ffill', inplace=True)
    raw_data.fillna(method='bfill', inplace=True)
    raw_data.replace([np.inf, -np.inf], -1.0)   # not sure whether it is appropriate.
    # 构建额外feature
    features_for_training = dict()
    ######################## price #######################
    # mean price of feature 1,2,3,4 
    raw_data['PriceFeature_mean']=raw_data[['PriceFeature_1','PriceFeature_2',
                            'PriceFeature_3','PriceFeature_4']].mean(axis=1)  # 4者平均数
    # bband ***
    bb_up, bb_mid, bb_low = talib.BBANDS(raw_data['PriceFeature_mean'], timeperiod=ROLLING_WINDOW)
    raw_data['new_price_bb_from_up'] = raw_data['PriceFeature_mean']/bb_up - 1
    features_for_training['new_price_bb_from_up'] = 'float64'
    raw_data['new_price_bb_from_low'] = raw_data['PriceFeature_mean']/bb_low - 1
    features_for_training['new_price_bb_from_low'] = 'float64'
    # gap between the price_feature 1 to 4
    raw_data['new_price_gap_1m2'] = (raw_data['PriceFeature_1']-raw_data['PriceFeature_2'])/raw_data['PriceFeature_mean']
    raw_data['new_price_gap_1m3'] = (raw_data['PriceFeature_1']-raw_data['PriceFeature_3'])/raw_data['PriceFeature_mean']
    raw_data['new_price_gap_1m4'] = (raw_data['PriceFeature_1']-raw_data['PriceFeature_4'])/raw_data['PriceFeature_mean']
    raw_data['new_price_gap_2m3'] = (raw_data['PriceFeature_2']-raw_data['PriceFeature_3'])/raw_data['PriceFeature_mean']
    raw_data['new_price_gap_2m4'] = (raw_data['PriceFeature_2']-raw_data['PriceFeature_4'])/raw_data['PriceFeature_mean']
    raw_data['new_price_gap_3m4'] = (raw_data['PriceFeature_3']-raw_data['PriceFeature_4'])/raw_data['PriceFeature_mean']
    # features_for_training['new_price_bb_from_low'] = 'float64'
    # raw_data['new_price_bb_exceed_up'] = raw_data['PriceFeature_mean'] > bb_up
    # features_for_training['new_price_bb_exceed_up'] = 'bool'
    # raw_data['new_price_bb_below_low'] = raw_data['PriceFeature_mean'] < bb_low
    # features_for_training['new_price_bb_below_low'] = 'bool'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_price_bb_from_up_t{i}'] = raw_data['new_price_bb_from_up'].shift(i)
        features_for_training[f'new_price_bb_from_up_t{i}'] = 'float64'
        raw_data[f'new_price_bb_from_low_t{i}'] = raw_data['new_price_bb_from_low'].shift(i)
        features_for_training[f'new_price_bb_from_low_t{i}'] = 'float64'

        raw_data[f'new_price_gap_1m2_t{i}'] = raw_data['new_price_gap_1m2'].shift(i)
        features_for_training[f'new_price_gap_1m2_t{i}'] = 'float64'
        raw_data[f'new_price_gap_1m3_t{i}'] = raw_data['new_price_gap_1m3'].shift(i)
        features_for_training[f'new_price_gap_1m3_t{i}'] = 'float64'
        raw_data[f'new_price_gap_1m4_t{i}'] = raw_data['new_price_gap_1m4'].shift(i)
        features_for_training[f'new_price_gap_1m4_t{i}'] = 'float64'
        raw_data[f'new_price_gap_2m3_t{i}'] = raw_data['new_price_gap_2m3'].shift(i)
        features_for_training[f'new_price_gap_2m3_t{i}'] = 'float64'
        raw_data[f'new_price_gap_2m4_t{i}'] = raw_data['new_price_gap_2m4'].shift(i)
        features_for_training[f'new_price_gap_2m4_t{i}'] = 'float64'
        raw_data[f'new_price_gap_3m4_t{i}'] = raw_data['new_price_gap_3m4'].shift(i)
        features_for_training[f'new_price_gap_3m4_t{i}'] = 'float64'
        # raw_data[f'new_price_bb_exceed_up_t{i}'] = raw_data['new_price_bb_exceed_up'].shift(i)
        # features_for_training[f'new_price_bb_exceed_up_t{i}'] = 'bool'
        # raw_data[f'new_price_bb_below_low_t{i}'] = raw_data['new_price_bb_below_low'].shift(i)
        # features_for_training[f'new_price_bb_below_low_t{i}'] = 'bool'
    # price momentum ***
    for i in LOOK_BACK_PERIOD:
        if i == 1:
            continue  # it's just one period return
        raw_data[f'new_price_mom_t{i}'] = raw_data['PriceFeature_mean'].pct_change(periods=i)
        features_for_training[f'new_price_mom_t{i}'] = 'float64'
    # log return ***
    raw_data['new_price_log_rt'] = np.log(raw_data['PriceFeature_mean']).diff()
    features_for_training['new_price_log_rt'] = 'float64'
    ## mean price change and look back ***
    ### log return look back ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_log_return_t{i}'] = raw_data['new_price_log_rt'].shift(i)
        features_for_training[f'new_log_return_t{i}'] = 'float64'
    ### PriceFeature_5 look back ***
    features_for_training['PriceFeature_5'] = 'float64'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_price_5_t{i}'] = raw_data['PriceFeature_5'].shift(i)
        features_for_training[f'new_price_5_t{i}'] = 'float64'
    ### PriceFeature_6 look back ***
    features_for_training['PriceFeature_6'] = 'float64'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_price_6_t{i}'] = raw_data['PriceFeature_6'].shift(i)
        features_for_training[f'new_price_6_t{i}'] = 'float64'
    ### PriceFeature_7 look back ***
    features_for_training['PriceFeature_7'] = 'bool'
    for i in LOOK_BACK_PERIOD :
        raw_data[f'new_price_7_t{i}'] = raw_data['PriceFeature_7'].shift(i)
        features_for_training[f'new_price_7_t{i}'] = 'float64'
    ### PriceFeature_8 look back ***
    features_for_training['PriceFeature_8'] = 'bool'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_price_8_t{i}'] = raw_data['PriceFeature_8'].shift(i)
        features_for_training[f'new_price_8_t{i}'] = 'float64'
    
    #TODO: 是否要考虑4个价格的差？？？这些差该如何做标准化呢？

    ######################## volume #######################
    raw_data['new_vol_1_rolling_mean'] = raw_data['VolumeFeature_1'].rolling(ROLLING_WINDOW).mean() + 0.0001
    # *** vol norm + look back
    raw_data['new_vol_1_norm'] = raw_data['VolumeFeature_1']/raw_data['new_vol_1_rolling_mean'] 
    features_for_training['new_vol_1_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_1_mom_t{i}'] = raw_data['new_vol_1_norm'].diff(periods=i)
        features_for_training[f'new_vol_1_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_1_norm_t{i}'] = raw_data['new_vol_1_norm'].shift(i)
        features_for_training[f'new_vol_1_norm_t{i}'] = 'float64'

    features_for_training['VolumeFeature_2'] = 'bool'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_2_t{i}'] = raw_data['VolumeFeature_2'].shift(i)
        features_for_training[f'new_vol_2_t{i}'] = 'bool'

    raw_data['new_vol_3_rolling_mean'] = raw_data['VolumeFeature_3'].rolling(ROLLING_WINDOW).mean() + 0.0001
    # *** vol norm + look back
    raw_data['new_vol_3_norm'] = raw_data['VolumeFeature_3']/raw_data['new_vol_3_rolling_mean'] 
    features_for_training['new_vol_3_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_3_mom_t{i}'] = raw_data['new_vol_3_norm'].diff(periods=i)
        features_for_training[f'new_vol_3_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_3_norm_t{i}'] = raw_data['new_vol_3_norm'].shift(i)
        features_for_training[f'new_vol_3_norm_t{i}'] = 'float64'

    raw_data['new_vol_4_rolling_mean'] = raw_data['VolumeFeature_4'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_4_norm'] = raw_data['VolumeFeature_4']/raw_data['new_vol_4_rolling_mean']
    features_for_training['new_vol_4_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_4_mom_t{i}'] = raw_data['new_vol_4_norm'].diff(periods=i)
        features_for_training[f'new_vol_4_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_4_norm_t{i}'] = raw_data['new_vol_4_norm'].shift(i)
        features_for_training[f'new_vol_4_norm_t{i}'] = 'float64'
        

    raw_data['new_vol_5_rolling_mean'] = raw_data['VolumeFeature_5'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_5_norm'] = raw_data['VolumeFeature_5']/raw_data['new_vol_5_rolling_mean']
    features_for_training['new_vol_5_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_5_mom_t{i}'] = raw_data['new_vol_5_norm'].diff(periods=i)
        features_for_training[f'new_vol_5_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_5_norm_t{i}'] = raw_data['new_vol_5_norm'].shift(i)
        features_for_training[f'new_vol_5_norm_t{i}'] = 'float64'

    raw_data['new_vol_6_rolling_mean'] = raw_data['VolumeFeature_6'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_6_norm'] = raw_data['VolumeFeature_6']/raw_data['new_vol_6_rolling_mean']
    features_for_training['new_vol_6_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_6_mom_t{i}'] = raw_data['new_vol_6_norm'].diff(periods=i)
        features_for_training[f'new_vol_6_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_6_norm_t{i}'] = raw_data['new_vol_6_norm'].shift(i)
        features_for_training[f'new_vol_6_norm_t{i}'] = 'float64'

    raw_data['new_vol_7_rolling_mean'] = raw_data['VolumeFeature_7'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_7_norm'] = raw_data['VolumeFeature_7']/raw_data['new_vol_7_rolling_mean']
    features_for_training['new_vol_7_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_7_mom_t{i}'] = raw_data['new_vol_7_norm'].diff(periods=i)
        features_for_training[f'new_vol_7_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_7_norm_t{i}'] = raw_data['new_vol_7_norm'].shift(i)
        features_for_training[f'new_vol_7_norm_t{i}'] = 'float64'

    raw_data['new_vol_8_rolling_mean'] = raw_data['VolumeFeature_8'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_8_norm'] = raw_data['VolumeFeature_8']/raw_data['new_vol_8_rolling_mean']
    features_for_training['new_vol_8_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_8_mom_t{i}'] = raw_data['new_vol_8_norm'].diff(periods=i)
        features_for_training[f'new_vol_8_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_8_norm_t{i}'] = raw_data['new_vol_8_norm'].shift(i)
        features_for_training[f'new_vol_8_norm_t{i}'] = 'float64'

    raw_data['new_vol_9_rolling_mean'] = raw_data['VolumeFeature_9'].rolling(ROLLING_WINDOW).mean() + 0.0001 # mean can be 0!
    raw_data['new_vol_9_norm'] = raw_data['VolumeFeature_9']/raw_data['new_vol_9_rolling_mean']
    features_for_training['new_vol_9_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_9_mom_t{i}'] = raw_data['new_vol_9_norm'].diff(periods=i)
        features_for_training[f'new_vol_9_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_9_norm_t{i}'] = raw_data['new_vol_9_norm'].shift(i)
        features_for_training[f'new_vol_9_norm_t{i}'] = 'float64'

    raw_data['new_vol_10_rolling_mean'] = raw_data['VolumeFeature_10'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_10_norm'] = raw_data['VolumeFeature_10']/raw_data['new_vol_10_rolling_mean']
    features_for_training['new_vol_10_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_10_mom_t{i}'] = raw_data['new_vol_10_norm'].diff(periods=i)
        features_for_training[f'new_vol_10_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_10_norm_t{i}'] = raw_data['new_vol_10_norm'].shift(i)
        features_for_training[f'new_vol_10_norm_t{i}'] = 'float64'

    raw_data['new_vol_11_rolling_mean'] = raw_data['VolumeFeature_11'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_11_norm'] = raw_data['VolumeFeature_11']/raw_data['new_vol_11_rolling_mean']
    features_for_training['new_vol_11_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_11_mom_t{i}'] = raw_data['new_vol_11_norm'].diff(periods=i)
        features_for_training[f'new_vol_11_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_11_norm_t{i}'] = raw_data['new_vol_11_norm'].shift(i)
        features_for_training[f'new_vol_11_norm_t{i}'] = 'float64'

    raw_data['new_vol_12_rolling_mean'] = raw_data['VolumeFeature_12'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_vol_12_norm'] = raw_data['VolumeFeature_12']/raw_data['new_vol_12_rolling_mean']
    features_for_training['new_vol_12_norm'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_12_mom_t{i}'] = raw_data['new_vol_12_norm'].diff(periods=i)
        features_for_training[f'new_vol_12_mom_t{i}'] = 'float64'
        raw_data[f'new_vol_12_norm_t{i}'] = raw_data['new_vol_12_norm'].shift(i)
        features_for_training[f'new_vol_12_norm_t{i}'] = 'float64'

    features_for_training['VolumeFeature_13'] = 'float64'
    ## vol Momentum ***
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_vol_13_t{i}'] = raw_data['VolumeFeature_13'].shift(i)
        features_for_training[f'new_vol_13_t{i}'] = 'float64'        

    ####################### Tape #######################
    features_for_training['TapeFeature_1'] = 'bool'
    features_for_training['TapeFeature_2'] = 'float64'
    features_for_training['TapeFeature_4'] = 'float64'
    features_for_training['TapeFeature_5'] = 'bool'
    features_for_training['TapeFeature_6'] = 'float64'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_tape_1_t{i}'] = raw_data['TapeFeature_1'].shift(i)
        features_for_training[f'new_tape_1_t{i}'] = 'bool'
        raw_data[f'new_tape_2_t{i}'] = raw_data['TapeFeature_2'].shift(i)
        features_for_training[f'new_tape_2_t{i}'] = 'float64'
        raw_data[f'new_tape_4_t{i}'] = raw_data['TapeFeature_4'].shift(i)
        features_for_training[f'new_tape_4_t{i}'] = 'float64'
        raw_data[f'new_tape_5_t{i}'] = raw_data['TapeFeature_5'].shift(i)
        features_for_training[f'new_tape_5_t{i}'] = 'bool'
        raw_data[f'new_tape_6_t{i}'] = raw_data['TapeFeature_6'].shift(i)
        features_for_training[f'new_tape_6_t{i}'] = 'float64'
    
    raw_data['new_tape_3_rolloing_abs_mean'] = raw_data['TapeFeature_3'].abs().rolling(ROLLING_WINDOW).mean() + 0.0001
    # normalize ***
    raw_data['new_tape_3_abs_norm'] = raw_data['TapeFeature_3']/raw_data['new_tape_3_rolloing_abs_mean'] 
    features_for_training['new_tape_3_abs_norm'] = 'float64'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_tape_3_abs_norm_t{i}'] = raw_data['new_tape_3_abs_norm'].shift(i)
        features_for_training[f'new_tape_3_abs_norm_t{i}'] = 'float64'

    raw_data['new_tape_3_rolloing_mean'] = raw_data['TapeFeature_3'].rolling(ROLLING_WINDOW).mean() + 0.0001
    # normalize ***
    raw_data['new_tape_3_norm'] = raw_data['TapeFeature_3']/raw_data['new_tape_3_rolloing_mean'] 
    features_for_training['new_tape_3_norm'] = 'float64'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_tape_3_norm_t{i}'] = raw_data['new_tape_3_abs_norm'].shift(i)
        features_for_training[f'new_tape_3_norm_t{i}'] = 'float64'    
    # TODO: sum the normalized tape in look back period

    ####################### Transaction #######################
    # look back
    features_for_training['TransactionFeature_1'] = 'bool'
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_trans_1_t{i}'] = raw_data['TransactionFeature_1'].shift(i)
        features_for_training[f'new_trans_1_t{i}'] = 'bool'

    ####################### order #######################
    # normalize 
    raw_data['new_order_1_rolloing_mean'] = raw_data['OrderFeature_1'].rolling(ROLLING_WINDOW).mean() + 0.0001
    raw_data['new_order_1_norm'] = raw_data['OrderFeature_1']/raw_data['new_order_1_rolloing_mean'] 
    features_for_training['new_order_1_norm'] = 'float64' 
    features_for_training['OrderFeature_2'] = 'float64' 
    features_for_training['OrderFeature_3'] = 'float64' 
    features_for_training['OrderFeature_4'] = 'float64' 
    for i in LOOK_BACK_PERIOD:
        raw_data[f'new_order_1_norm_t{i}'] = raw_data['new_order_1_norm'].shift(i)
        features_for_training[f'new_order_1_norm_t{i}'] = 'float64'
        raw_data[f'new_order_2_t{i}'] = raw_data['OrderFeature_2'].shift(i)
        features_for_training[f'new_order_2_t{i}'] = 'float64'
        raw_data[f'new_order_3_t{i}'] = raw_data['OrderFeature_3'].shift(i)
        features_for_training[f'new_order_3_t{i}'] = 'float64'
        raw_data[f'new_order_4_t{i}'] = raw_data['OrderFeature_4'].shift(i)
        features_for_training[f'new_order_4_t{i}'] = 'float64'
    return raw_data, features_for_training

def sampling(raw_data):
    # ----------------------- sampling --------------------------------
    df = raw_data.dropna()
    
    first_index = df.iloc[[0,-1]].index[0]
    last_index = df.iloc[[0,-1]].index[1]
    sample_ids = set()
    # keep all TargetFeature = 1
    for i in df[df['TargetFeature']==1].index:
        sample_ids.add(i)
        # also get the neighbor
        if i - 1 >= first_index:
            sample_ids.add(i - 1)
        if i + 1 <= last_index:
            sample_ids.add(i + 1)
    row_index = first_index - SAMPLE_INTERVAL[0]  # in accordance with randrange(3, 15)
    while True:
        row_index += randrange(SAMPLE_INTERVAL[0], SAMPLE_INTERVAL[1])  # uniform distribution between 3~15
        if row_index <= last_index:
            sample_ids.add(row_index)
        else:
            break
    sample_ids = sorted(sample_ids)
    return df.reindex(sample_ids)


    # randrange(3, 15)
    

############# main################
total_data = pd.DataFrame()
for label in LABELS:
    print(f'processing {label} file ......')
    data, features = preprocess_data(label)
    sample_data = sampling(data)
    sample_data['label'] = label  # add a new column/feature
    total_data = total_data.append(sample_data.copy())    
total_data.replace([np.inf, -np.inf], -1.0, inplace=True)

#-------------- feature selection
# use DecisionTreeClassifier
df_feature_sel = total_data[total_data["label"] == 'Stock_1']
X = df_feature_sel[list(features.keys())]
# X = X.replace([np.inf, -np.inf], -1.0)   # ???How to handle inf??? # !!! to be deleted
y = df_feature_sel['TargetFeature']
clf = DecisionTreeClassifier(class_weight='balanced')  # consider the imbalance training data
trans = SelectFromModel(clf)  #, threshold='0.1*mean'
X_trans = trans.fit_transform(X, y)
print("We started with {0} features but retained only {1} of them!".format(X.shape[1], X_trans.shape[1]))
useful_featrues_1 = X.columns[trans.get_support()].values

# use RandomForestClassifier
le = LabelEncoder()
# X = total_data.replace([np.inf, -np.inf], -1.0) # !!! to be deleted
total_data["label_encode"] = le.fit_transform(total_data["label"].values)
feature_list = list(features.keys())
feature_list.append('label_encode')  # Stock_1 ... 10 is a feature.
X = total_data[feature_list]
y = total_data['TargetFeature']
clf = RandomForestClassifier(n_estimators=150, random_state=0, class_weight='balanced')
trans = SelectFromModel(clf, threshold='median') 
X_trans = trans.fit_transform(X, y)
print("We started with {0} features but retained only {1} of them!".format(X.shape[1], X_trans.shape[1]))
useful_featrues_2 = X.columns[trans.get_support()].values
# merge the features 
features_for_train = set(useful_featrues_1) | set(useful_featrues_2)
features_for_train.add('label_encode')
features_for_train = list(features_for_train)

### 训练模型
X_train, X_validate, y_train, y_validate = train_test_split(total_data[features_for_train], 
            y, test_size=0.3, shuffle=True) 
parameters = {'max_depth':[3, 4, 5, 6, 7, 8],
              'n_estimators':[50, 100, 256, 512],
              'criterion':['gini', 'entropy'],
              'random_state':[42]}

def perform_grid_search(X_data, y_data):
    rf = RandomForestClassifier(class_weight='balanced')
    clf = GridSearchCV(rf, parameters, scoring='roc_auc', n_jobs=3)  # cv=4, 
    clf.fit(X_data, y_data)
    print(clf.cv_results_['mean_test_score'])
    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']

n_estimator, depth = perform_grid_search(X_train, y_train)
c_random_state = 42
print(n_estimator, depth, c_random_state)



def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

# get na summary 
na_sum = dict()
for label in LABELS:
    file_path = os.path.join(DATA_HOME, f'{label}.csv')
    raw_data = pd.read_csv(file_path, index_col=0)
    na_sum[label] = raw_data.isna().sum()

na_sum_df = pd.DataFrame(na_sum)


# some sample code 
# feat_labels = data.columns[1:]
# clf = RandomForestClassifier(n_estimators=100, random_state=0)

# # Train the classifier
# clf.fit(X_train, y_train)

# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]

# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))  

# # Create a selector object that will use the random forest classifier to identify
# # features that have an importance of more than 0.15
# sfm = SelectFromModel(clf, threshold=0.15)

# # Train the selector
# sfm.fit(X_train, y_train)