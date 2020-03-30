import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
import multiprocessing as mp
import datetime as dt
from IPython.display import display
from IPython.core.debugger import set_trace as bp
from pathlib import PurePath, Path
from multiprocessing import cpu_count
import sys
import time
from collections import OrderedDict as od
import re
import os
import json
from src.utils.utils import *
import src.features.bars as brs
import src.features.snippets as snp


def getTEvents(gRaw, h):
    """
    gRaw - 价格序列
    h - 阀值
    当 gRaw 累计超过h时，被当作一种event。
    返回时间发生的时间点。
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break
        sPos, sNeg=max(0., pos), min(0., neg)
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def getDailyVol(close,span0=100):
    # daily vol reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]   
    df0=pd.Series(close.index[df0-1], 
                   index=close.index[close.shape[0]-df0.shape[0]:])
    try:
        df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0

def applyPtSlOnT1(close,events,ptSl,molecule):
    """
    apply stop loss/profit taking, if it takes place before t1 (end of event)

    close: A pandas series of prices
    events: A pandas dataframe, with columns,
        ◦ t1: The timestamp of vertical barrier. When the value is np.nan, there will
            not be a vertical barrier.
        ◦ trgt: The unit width of the horizontal barriers
    ptSl: A list of two non-negative ﬂoat values:
        ◦ ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
            If 0, there will not be an upper barrier.
        ◦ ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
            If 0, there will not be a lower barrier.
    molecule: A list with the subset of event indices that will be processed by a single thread. 
    """
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0: 
        pt=ptSl[0]*events_['trgt']
    else: 
        pt=pd.Series(index=events.index) # NaNs

    if ptSl[1]>0: 
        sl=-ptSl[1]*events_['trgt']
    else: 
        sl=pd.Series(index=events.index) # NaNs

    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
    return out  

def addVerticalBarrier(tEvents, close, numDays=1):
    """
    找到numDays后对应在tEvents的index。
    """
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1

def getBinsOld(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except:
        pass
    return out

def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    
    output:
    ret: The return realized at the time of the frst touched barrier.
    bin: The label, {−1, 0, 1}, as a function of the sign of the outcome. The function
        can be easily adjusted to label as 0 those events when the vertical barrier was
        touched frst, which we leave as an exercise.
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')  # reindex -- sort the df according to the new index
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

def getBinsNew(events, close, t1=None):
    '''
    [3.3] Adjust the getBins function to return a 0 whenever the vertical barrier is the one touched first

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    # 2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    
    if 'side' not in events_:
        # only applies when not meta-labeling
        # to update bin to 0 when vertical barrier is touched, we need the original
        # vertical barrier series since the events['t1'] is the time of first 
        # touch of any barrier and not the vertical barrier specifically. 
        # The index of the intersection of the vertical barrier values and the 
        # events['t1'] values indicate which bin labels needs to be turned to 0
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.
    
    if 'side' in events_:
        out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out    

def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:
            break
        print('dropped label: ', df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events

def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    calls a multiprocessing engine

    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    
    if linMols:
        parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:
        parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:
        out=processJobs_(jobs)
    else: 
        out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):
        df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):
        df0=pd.Series()
    else:
        return out
    for i in out:
        df0=df0.append(i)
    df0=df0.sort_index()
    return df0

def processJobs_(jobs):
    """single-thread execution for debugging [20.8]"""
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    finds the time of the first barrier touch. 

    close: close prices time series
    tEvents: 前置（条件）event发生的时间序列。
    ptSl: profit take / stop loss. 设置此两条线的宽度, multiplies trgt to get the width of upper and lower
        barries
    t1: vertical barriers
    trgt: target return
    minRet: minimum target return 
    side: 操作方向 空/多 -1/1
    """
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:
        t1=pd.Series(pd.NaT, index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:
        # 不知道side的情况下，上下barries必须对称
        side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: 
        side_,ptSl_=side.loc[trgt.index],ptSl[:2]

    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna(subset=['trgt'])) # 不满足target return的 event 全部剔除。

    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
                    numThreads=numThreads,close=close,events=events,
                    ptSl=ptSl_)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:
        events=events.drop('side',axis=1)
    return events

def get_up_cross(df):
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return df.fast[(crit1) & (crit2)]

def get_down_cross(df):
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return df.fast[(crit1) & (crit2)]

def bbands(price, window=None, width=None, numsd=None):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)        
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)

def get_up_cross_bb(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper.shift(1)  
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def get_down_cross_bb(df, col):
    # col is price column    
    crit1 = df[col].shift(1) > df.lower.shift(1) 
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def df_rolling_autocorr(df, window, lag=1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""

    return (df.rolling(window=window)
            .corr(df.shift(lag))) # could .dropna() here    

def resample_tick_data(df, rule="1S"):
    """ 
        default resample to 1 seconds
        columns:
        price -- keep last value
        bid -- keep last value
        ask -- keep last value
        size -- sum
        v -- sum
        dv -- sum    
    """
    print("start processing price...")
    price = df['price'].resample(rule).last()
    price = price.dropna()
    print("start processing bid...")
    bid = df['bid'].resample(rule).last()
    bid = bid.dropna()
    print("start processing ask...")
    ask = df['ask'].resample(rule).last()
    ask = ask.dropna()
    print("start processing size...")
    size = df['size'].resample(rule).sum()
    size = size.dropna()
    print("start processing v...")
    v = df['v'].resample(rule).sum()
    v = v.dropna()
    print("start processing dv...")
    dv = df['dv'].resample(rule).sum()
    dv = dv.dropna()
    print('create a dict...')
    resampled_df = {
                        'price': price,
                        'bid':bid,
                        'ask':ask,
                        'size':size,
                        'v':v,
                        'dv':dv
                    }
    print("convert to dataframe...")
    resampled_df = pd.DataFrame(resampled_df).dropna()
    return resampled_df


if __name__ == "__main__":
    ## (a) Run cusum filter with threshold equal to std dev of daily returns
    import os
    # load data
    data_root = "C:\\myproject\\advances_in_fin_ml\\Adv_Fin_ML_Exercises\\data\\processed"
    resampled_0 = os.path.join(data_root, "resampled0.parq")
    resampled_1 = os.path.join(data_root, "resampled1.parq")
    resampled_2 = os.path.join(data_root, "resampled2.parq")
    # tick_file = "C:\\myproject\\advances_in_fin_ml\\Adv_Fin_ML_Exercises\\data\\processed\\clean_IVE_fut_prices_resampled.parq"
    df = pd.read_parquet(resampled_0)
    df = df.append(pd.read_parquet(resampled_1))
    df = df.append(pd.read_parquet(resampled_2))
    # resample to 1 second. 
    # 因为，如果不做resample的话，那么tick的timestamp是有可能重复的。比如某个时间点上有多个成交
    # 就会导致以tick 的 timestamp作为index是不唯一的。造成getDailyVol失败。
    # df = resample_tick_data(df)
    # save the resampled file
    # tick_file = "C:\\myproject\\advances_in_fin_ml\\Adv_Fin_ML_Exercises\\data\\processed\\clean_IVE_fut_prices_resampled.parq"
    # df.to_parquet(tick_file)

    # convert to dollar bar.
    # dbars = brs.dollar_bar_df(df, 'dv', 1_000_000).drop_duplicates().dropna()
    dbars = brs.dollar_bar_df(df, 'dv', 500_000).drop_duplicates().dropna()
    cprint(dbars)
    # get the colose price
    close = dbars.price.copy()
    # get rolling STD
    dailyVol = getDailyVol(close)
    cprint(dailyVol.to_frame())

    f,ax=plt.subplots()
    dailyVol.plot(ax=ax)
    ax.axhline(dailyVol.mean(),ls='--',color='r')
    # tEvents = getTEvents(close,h=dailyVol.mean())
    # tEvents

    ## (b) Add vertical barrier
    # t1 = addVerticalBarrier(tEvents, close, numDays=1)
    # t1  

    # ## (c) Apply triple-barrier method where ptSl = [1,1] and t1 is the series 
    # # create target series
    # ptsl = [1,1]
    # target=dailyVol
    # # select minRet
    # minRet = 0.01

    # # Run in single-threaded mode on Windows
    import platform
    if platform.system() == "Windows":
        cpus = 1
    else:
        cpus = cpu_count() - 1
  
    # events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
    # cprint(events)

    # # (d) Apply getBins to generate labels
    # labels = getBins(events, close)
    # cprint(labels)
    # print(labels.bin.value_counts())

    # # [3.2] Use snippet 3.8 to drop under-populated labels
    # clean_labels = dropLabels(labels)
    # cprint(clean_labels)

    ###### [3.4] Develop moving average crossover strategy. For each obs. the model suggests a side but not size of the bet
    # fast_window = 3
    # slow_window = 7
    ## hyper param!!!
    fast_window = 5
    slow_window = 30

    close_df = (pd.DataFrame()
            .assign(price=close)
            .assign(fast=close.ewm(fast_window).mean())
            .assign(slow=close.ewm(slow_window).mean()))
    cprint(close_df)

    #### 以下找金叉死叉的timestamp作为side的结果是不合逻辑的
    #### 因为金叉死叉发生的timestamp和event发生的timestamp明显是相互独立的
    #### 因此一个比较正确的设置side的方式是，如果fast>slow 则是long，反正则是short

    # up = get_up_cross(close_df) 
    # down = get_down_cross(close_df)

    # f, ax = plt.subplots(figsize=(11,8))

    # close_df.loc['2010':].plot(ax=ax, alpha=.5)
    # up.loc['2010':].plot(ax=ax,ls='',marker='^', markersize=7,
    #                     alpha=0.75, label='upcross', color='g')
    # down.loc['2010':].plot(ax=ax,ls='',marker='v', markersize=7, 
    #                     alpha=0.75, label='downcross', color='r')

    # ax.legend()

    # # (a) Derive meta-labels for ptSl = [1,2] and t1 where numdays=1. Use as trgt dailyVol 
    # # computed by snippet 3.1 (get events with sides)
    # side_up = pd.Series(1, index=up.index)
    # side_down = pd.Series(-1, index=down.index)
    # side = pd.concat([side_up,side_down]).sort_index()
    # cprint(side)
    #######################################################

    long_signals = close_df['fast'] >= close_df['slow'] 
    short_signals = close_df['fast'] < close_df['slow']  
    close_df.loc[long_signals, 'side'] = 1
    close_df.loc[short_signals, 'side'] = -1
    close_df['side'] = close_df['side'].shift(1)
    # Remove Look ahead biase by lagging the signal
    close_df['side'] = close_df['side'].shift(1)
    cprint(close_df['side'])

    # Save the raw data
    raw_data = close_df.copy()

    # Drop the NaN values from our data set
    close_df.dropna(axis=0, how='any', inplace=True)
    close_df['side'].value_counts()


    minRet = 0.005 
    ptsl=[1,2]

    dailyVol = getDailyVol(close_df['price'])
    tEvents = getTEvents(close_df['price'],h=dailyVol.mean())
    t1 = addVerticalBarrier(tEvents, close_df['price'], numDays=1)

    # get the time for the first touch.
    # t1 被修改，从vertical bar --> first touch timestamp.
    ma_events = getEvents(
                            close = close_df['price'],
                            tEvents = tEvents,
                            ptSl = ptsl,
                            trgt = dailyVol,
                            minRet = minRet,
                            numThreads = cpus,
                            t1=t1,
                            side=close_df['side'])
    cprint(ma_events)
    ma_events.side.value_counts()
    ma_side = ma_events.dropna().side
    # 如果有side的info的情况下 label 0 or 1，交易或不交易！
    # 否则，label -1 or 1，只代表触碰上沿或下沿--交易方向（side）
    # index 是event_t timestamp.
    # 如果输入side，那么在getBins里面还需要判断触碰上下沿同side的方向是否一致，
    # 也就是return是否为正。排除如：触碰上沿但是side是做空的情况。
    ma_bins = getBinsNew(ma_events,close_df['price'], t1).dropna()
    ma_bins.bin.value_counts()
    cprint(ma_bins)

    # ### Fit a Meta Model
    # Train a random forest to decide whether to trade or not (i.e 1 or 0 respectively) since the earlier model has decided the side (-1 or 1)

    # Create the following features:

    # Volatility
    # Serial Correlation
    # The returns at the different lags from the serial correlation
    # The sides from the SMavg Strategy
    # 以上这些feature是决定分类器performance的最关键的地方！！！
    # 下面的这些feature只是演示作用

    raw_data['log_ret'] = np.log(raw_data['price']).diff()
    # Momentum
    raw_data['mom1'] = raw_data['price'].pct_change(periods=1)
    raw_data['mom2'] = raw_data['price'].pct_change(periods=2)
    raw_data['mom3'] = raw_data['price'].pct_change(periods=3)
    raw_data['mom4'] = raw_data['price'].pct_change(periods=4)
    raw_data['mom5'] = raw_data['price'].pct_change(periods=5)
    # Volatility
    raw_data['volatility_50'] = raw_data['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    raw_data['volatility_31'] = raw_data['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    raw_data['volatility_15'] = raw_data['log_ret'].rolling(window=15, min_periods=15, center=False).std()
    # Serial Correlation (Takes about 4 minutes)
    window_autocorr = 50

    raw_data['autocorr_1'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    raw_data['autocorr_2'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    raw_data['autocorr_3'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    raw_data['autocorr_4'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    raw_data['autocorr_5'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)
    # Get the various log -t returns
    raw_data['log_t1'] = raw_data['log_ret'].shift(1)
    raw_data['log_t2'] = raw_data['log_ret'].shift(2)
    raw_data['log_t3'] = raw_data['log_ret'].shift(3)
    raw_data['log_t4'] = raw_data['log_ret'].shift(4)
    raw_data['log_t5'] = raw_data['log_ret'].shift(5)
    # Re compute sides
    raw_data['side'] = np.nan

    long_signals = raw_data['fast'] >= raw_data['slow']
    short_signals = raw_data['fast'] < raw_data['slow']

    raw_data.loc[long_signals, 'side'] = 1
    raw_data.loc[short_signals, 'side'] = -1

    # Remove look ahead bias
    raw_data = raw_data.shift(1)

    # Get features at event dates
    X = raw_data.loc[ma_bins.index, :]
    X.drop(['price', 'fast', 'slow',], axis=1, inplace=True)
    cprint(X)

    # 这里显示了t0 event发生后，能产生盈利交易和无法盈利交易的数据
    y = ma_bins['bin']
    y.value_counts()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score
    from sklearn.utils import resample
    from sklearn.utils import shuffle

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    # Split data into training, validation and test sets
    X_training_validation = X['2009-10-02':'2017-10-01']
    y_training_validation = y['2009-10-02':'2017-10-01']
    X_train, X_validate, y_train, y_validate = train_test_split(X_training_validation, y_training_validation, test_size=0.15, shuffle=False)

    train_df = pd.concat([y_train, X_train], axis=1, join='inner')
    train_df['bin'].value_counts()

    # Upsample the training data to have a 50 - 50 split
    # https://elitedatascience.com/imbalanced-classes
    majority = train_df[train_df['bin'] == 1]
    minority = train_df[train_df['bin'] == 0]

    new_minority = resample(minority, 
                    replace=True,     # sample with replacement
                    n_samples=majority.shape[0],    # to match majority class
                    random_state=42)

    train_df = pd.concat([majority, new_minority])
    train_df = shuffle(train_df, random_state=42)

    train_df['bin'].value_counts()    

    # Create training data
    y_train = train_df['bin']
    X_train= train_df.loc[:, train_df.columns != 'bin']

    ### Fit a model
    parameters = {'max_depth':[2, 3, 4, 5, 7],
                'n_estimators':[1, 10, 25, 50, 100, 256, 512],
                'random_state':[42]}
        
    def perform_grid_search(X_data, y_data):
        rf = RandomForestClassifier(criterion='entropy')
        
        clf = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)
        
        clf.fit(X_data, y_data)
        
        print(clf.cv_results_['mean_test_score'])
        
        return clf.best_params_['n_estimators'], clf.best_params_['max_depth']

    # extract parameters
    n_estimator, depth = perform_grid_search(X_train, y_train)
    c_random_state = 42
    print(n_estimator, "_", depth, "_", c_random_state)

    # Refit a new model with best params, so we can see feature importance
    rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                                criterion='entropy', random_state=c_random_state)

    rf.fit(X_train, y_train.values.ravel())

    ### Training Metrics --- 
    # Performance Metrics
    y_pred_rf = rf.predict_proba(X_train)[:, 1]
    y_pred = rf.predict(X_train)
    fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
    print(classification_report(y_train, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_train, y_pred))

    print('')
    print("Accuracy")
    print(accuracy_score(y_train, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    ### Validation Metrics
    # Meta-label
    # Performance Metrics
    y_pred_rf = rf.predict_proba(X_validate)[:, 1]
    y_pred = rf.predict(X_validate)
    fpr_rf, tpr_rf, _ = roc_curve(y_validate, y_pred_rf)
    print(classification_report(y_validate, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_validate, y_pred))

    print('')
    print("Accuracy")
    print(accuracy_score(y_validate, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    print(X_validate.index.min())
    print(X_validate.index.max())

    # Feature Importance
    title = 'Feature Importance:'
    figsize = (15, 5)

    feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})    
    feat_imp['feature'] = X.columns
    feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
    feat_imp = feat_imp

    feat_imp.sort_values(by='Importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()

    ### Perform out-of-sample test
    #   Meta Model Metrics
    # Extarct data for out-of-sample (OOS)
    X_oos = X['2017-10-02':]
    y_oos = y['2017-10-02':]

    # Performance Metrics
    y_pred_rf = rf.predict_proba(X_oos)[:, 1]
    y_pred = rf.predict(X_oos)
    fpr_rf, tpr_rf, _ = roc_curve(y_oos, y_pred_rf)
    print(classification_report(y_oos, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_oos, y_pred))

    print('')
    print("Accuracy")
    print(accuracy_score(y_oos, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    ###########################END#################################################################################
    ############################################################################################################
    # Xx = pd.merge_asof(ma_bins, side.to_frame().rename(columns={0:'side'}),
    #                left_index=True, right_index=True, direction='forward')
    # cprint(Xx)

    # ##### (b) Train Random Forest to decide whether to trade or not {0,1} 
    # # since underlying model (crossing m.a.) has decided the side, {-1,1}

    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import roc_curve, classification_report

    # X = ma_side.values.reshape(-1,1)
    # #X = Xx.side.values.reshape(-1,1)
    # y = ma_bins.bin.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
    #                                                     shuffle=False)

    # n_estimator = 10000
    # RANDOM_STATE = 777
    # rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
    #                             criterion='entropy', random_state=RANDOM_STATE)
    # rf.fit(X_train, y_train)

    # # The random forest model by itself
    # y_pred_rf = rf.predict_proba(X_test)[:, 1]
    # y_pred = rf.predict(X_test)
    # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    # print(classification_report(y_test, y_pred))

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    # ### [3.5] Develop mean-reverting Bollinger Band Strategy. For each obs. model suggests 
    # # a side but not size of the bet.
    # window=50
    # bb_df = pd.DataFrame()
    # bb_df['price'],bb_df['ave'],bb_df['upper'],bb_df['lower']=bbands(close, window=window, numsd=1)
    # bb_df.dropna(inplace=True)
    # cprint(bb_df)

    # f,ax=plt.subplots(figsize=(11,8))
    # bb_df.loc['2010'].plot(ax=ax)

    # bb_down = get_down_cross_bb(bb_df, 'price')
    # bb_up = get_up_cross_bb(bb_df, 'price') 

    # f, ax = plt.subplots(figsize=(11,8))

    # bb_df.loc['2010':].plot(ax=ax, alpha=.5)
    # bb_up.loc['2010':].plot(ax=ax, ls='', marker='^', markersize=7,
    #                         alpha=0.75, label='upcross', color='g')
    # bb_down.loc['2010':].plot(ax=ax, ls='', marker='v', markersize=7, 
    #                         alpha=0.75, label='downcross', color='r')
    # ax.legend()

    # # (a) Derive meta-labels for ptSl=[0,2] and t1 where numdays=1. Use as trgt dailyVol.
    # bb_side_up = pd.Series(-1, index=bb_up.index) # sell on up cross for mean reversion
    # bb_side_down = pd.Series(1, index=bb_down.index) # buy on down cross for mean reversion
    # bb_side_raw = pd.concat([bb_side_up,bb_side_down]).sort_index()
    # cprint(bb_side_raw)

    # minRet = .01 
    # ptsl=[0,2]
    # bb_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1,side=bb_side_raw)
    # cprint(bb_events)

    # bb_side = bb_events.dropna().side
    # cprint(bb_side)

    # bb_side.value_counts()

    # bb_bins = getBins(bb_events,close).dropna()
    # cprint(bb_bins)
    # print(bb_bins.bin.value_counts())

    # ## (b) train random forest to decide to trade or not. Use features: volatility, serial correlation,
    # #  and the crossing moving averages from exercise 2.
    # srl_corr = df_rolling_autocorr(returns(close), window=window).rename('srl_corr')
    # cprint(srl_corr)

    # features = (pd.DataFrame()
    #         .assign(vol=bb_events.trgt)
    #         .assign(ma_side=ma_side)
    #         .assign(srl_corr=srl_corr)
    #         .drop_duplicates()
    #         .dropna())
    # cprint(features)

    # Xy = (pd.merge_asof(features, bb_bins[['bin']], 
    #                 left_index=True, right_index=True, 
    #                 direction='forward').dropna())
    # cprint(Xy)
    # Xy.bin.value_counts()

    # X = Xy.drop('bin',axis=1).values
    # y = Xy['bin'].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
    #                                                 shuffle=False)
    # n_estimator = 10000
    # rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
    #                             criterion='entropy', random_state=RANDOM_STATE)
    # rf.fit(X_train, y_train)