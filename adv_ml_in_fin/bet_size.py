import numpy as np

from scipy.stats import norm, moment

import pandas as pd
from dask.diagnostics import ProgressBar
ProgressBar().register()  # ready progress bar once in notebook for later calculations
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import datetime as dt

# draw random numbers from a uniform distribution (all bets are long)
np.random.seed(0)
sample_size = 10_000
P_t = np.random.uniform(.5, 1., sample_size)  # array of random from uniform dist.

# 10.2(a) Compute bet sizes for ||X||=2
z = (P_t - 0.5) /  (P_t*(1-P_t))**0.5
m = 2 * norm.cdf(z) - 1  # bet sizes, x=1


# 10.2(b) assign 10,000 consecutive calendar days
start_date = dt.datetime(2000, 1, 1)  # starting at 01-JAN-2000
date_step = dt.timedelta(days=1)
dates = np.array([start_date + i*date_step for i in range(sample_size)])
bet_sizes = pd.Series(data=m, index=dates)


# 10.2(c) draw 10,000 random numbers from a uniform distribution
shift_list = np.random.uniform(1., 25., sample_size)
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])


# 10.2(d) create a pandas.Series object
dates_shifted = dates + shift_dt
t1 = pd.Series(data=dates_shifted, index=dates)

# Collect the series into a single DataFrame.
# Add a randomized 'side' indicator so we have both long and short bets.
df_events = pd.concat(objs=[t1, bet_sizes], axis=1)
df_events = df_events.rename(columns={0: 't1', 1: 'bet_size'})
df_events['p'] = P_t
df_events = df_events[['t1', 'p', 'bet_size']]


# 10.2(e) compute the average active bets
avg_bet = pd.Series()
active_bets = pd.Series()
for idx, val in t1.iteritems():
    active_idx = t1[(t1.index<=idx)&(t1>idx)].index
    num_active = len(active_idx)
    active_bets[idx] = num_active
    avg_bet[idx] = bet_sizes[active_idx].mean()

df_events['num_active_bets'] = active_bets
df_events['avg_active_bets'] = avg_bet


print("The first 10 rows of the resulting DataFrame from Exercise 10.2:")
display(df_events.head(10))
print("Summary statistics on the bet size columns:")
display(df_events[['bet_size', 'num_active_bets', 'avg_active_bets']].describe())