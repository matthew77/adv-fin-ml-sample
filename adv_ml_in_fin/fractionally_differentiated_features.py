import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# [5.1] Generate a time series from an IID Gaussian random process. 
# This is a memory-less, stationary series:

np.random.seed(0)

N = 252*10
s = pd.Series(np.random.randn(N))
s.plot()

# (a) Compute the ADF statistic on this series. What is the p-value?
adf = lambda s: sm.tsa.stattools.adfuller(s)
p_val = lambda s: sm.tsa.stattools.adfuller(s)[1]
res = adf(s); p = res[1]
res, p
# 以上p=0， 说明是stationary。

# (b) Compute the cumulative sum of the observations. 
# This is a non-stationary series w/o memory.
cmsm = pd.Series(s).cumsum()
cmsm.plot()

# (i) What is the order of integration of this cumulative series?
orders = [0, 1, 2, 3, 4]
for o in orders:
    diff_ = np.diff(cmsm,o)
    print('='*27)
    print(f'order: {o}, pVal: {p_val(diff_)}')

# (ii) Compute the ADF statistic on this series. What is the p-value?
p_val(cmsm)

# (c) Differentiate the series twice. What is the p-value of this 
# over-differentiated series?
diff_ = np.diff(cmsm,2)
p_val(diff_)

# [5.2] Generate a time series that follows a sinusoidal function. 
# This is a stationary series with memory.
np.random.seed(0)

rand = np.random.random(N)

idx = np.linspace(0,10, N)
s = pd.Series(1*np.sin(2.*idx + .5))
s.plot()

# (a) Compute the ADF statistic on this series. What is the p-value?
p_val(s)

# (b) Shift every observation by the same positive value. Compute the
#  cumulative sum of the observations. This is a non-stationary series with memory.
s_ = (s + 1).cumsum().rename('fake_close').to_frame()
s_.plot()

# (i) Compute the ADF statistic on this series. What is the p-value?
adf(s_['fake_close'].dropna()), p_val(s_['fake_close'])

# (ii) Apply an expanding window fracdiff, with  𝜏=1𝐸−2 . 
# For what minimum  𝑑  value do you get a p-value below  5% ?
def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w 

def fracDiff(series, d, thres=0.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily
        bounded between [0,1]
    '''
    #1) Compute weights for the longest series
    w=getWeights(d, series.shape[0])
    #bp()
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc=seriesF.index[iloc]
            test_val = series.loc[loc,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any(): 
                continue # exclude NAs
            try:
                df_.loc[loc]=np.dot(w[-(iloc+1):,:].T, seriesF.loc[:loc])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df    

cols = ['adfStat','pVal','lags','nObs','95% conf']  # ,'corr']
out = pd.DataFrame(columns=cols)
for d in np.linspace(0,1,11):
    try:
        df0 = fracDiff(s_,d)
        df0 = sm.tsa.stattools.adfuller(df0['fake_close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df0[:4])+[df0[4]['5%']]
    except: 
        break

f,ax=plt.subplots()
out['adfStat'].plot(ax=ax, marker='X')
ax.axhline(out['95% conf'].mean(),lw=1,color='r',ls='dotted')
ax.set_title('min d with thresh=0.01')
ax.set_xlabel('d values')
ax.set_ylabel('adf stat')
display(out)

# (iii) Apply FFD with  𝜏=1𝐸−5 . For what minimum  𝑑  value do you get
#  a p-value below  5%
cols = ['adfStat','pVal','lags','nObs','95% conf']#,'corr']
out = pd.DataFrame(columns=cols)
for d in np.linspace(0,1,11):
    try:
        df0 = fracDiff(s_,d,thres=1e-5)
        df0 = sm.tsa.stattools.adfuller(df0['fake_close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df0[:4])+[df0[4]['5%']]
    except Exception as e:
        print(f'd: {d}, error: {e}')
        continue

f,ax=plt.subplots()
out['adfStat'].plot(ax=ax, marker='X')
ax.axhline(out['95% conf'].mean(),lw=1,color='r',ls='dotted')
ax.set_title('min d with thresh=0.0001')
ax.set_xlabel('d values')
ax.set_ylabel('adf stat')
display(out)

# [5.3] Take the series from exercise 2.b:
# (a) Fit the series to a sine function. What is the R-squared?
## fitting function taken from stackoverflow
##   https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy#16716964
import numpy, scipy.optimize

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters 
    "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

res = fit_sin(s_.index.values, s_.values.ravel())
res

xx = s_.index.values
yy = s_.values.ravel()

plt.plot(xx, yy, "-k", label="y", linewidth=2)
#plt.plot(tt, yynoise, "ok", label="y with noise")
plt.plot(xx, res["fitfunc"](xx), "r-", label="y fit curve", linewidth=2)
plt.legend(loc="best")
plt.show()

# (b) Apply FFD (𝑑=1) . Fit the series to a sine function. What is
#  the R-squared?
df1 = fracDiff(s_,d=1)
df1.plot()

xx = df1.index.values
yy = df1.values.ravel()

res = fit_sin(xx, yy)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(yy, res["fitfunc"](xx))
r_value**2

# (c) What value of d maximizes the R-squared of a sinusoidal fit on FFD (𝑑) ? Why?