import os
os.chdir('C:\\Users\\nooree\\Google Drive\\Python\\PythonScratchpad\\first\\')

from numpy import genfromtxt
my_data = genfromtxt('.\\data\\NG12 G1 CD41.csv', delimiter=',')


def nantozero(myndarray): #replace nan values with zero
    myshapetuple = myndarray.shape
    xcount = 0
    ycount = 0
    for x in range(1,myshapetuple[0]):
        for y in range(1, myshapetuple[1]):
            if myndarray[x, y] == float('nan'):
                myndarray[x, y] = 0
            
    return myndarray[1:,1:]  
    
#check for NaNs, replace with zero and trim the header and
#time point row/column from the data    
ts_data = nantozero(my_data)

import numpy
ts_mean = numpy.mean(ts_data, axis = 1) #axis=1 gives the rowmeans
ts_std = numpy.std(ts_data, axis =1)
ts_med = numpy.median(ts_data, axis =1)
  
ts_minus = ts_mean - ts_std
ts_plus = ts_mean + ts_std
  
import matplotlib.pyplot
data_shape = my_data.shape
times = my_data[1:data_shape[0],0]
matplotlib.pyplot.plot(times, ts_plus, 'b-', times, ts_mean, 'r-', times, ts_minus, 'b-', times, ts_med, 'g-')
matplotlib.pyplot.xlabel('Time in seconds')
matplotlib.pyplot.ylabel('Intensity')
matplotlib.pyplot.title('Summary graph for NG12 G1 CD41')
matplotlib.pyplot.show()

import pandas as pd
#create a timeseries

pd_ts_mean = pd.Series(ts_mean, times)
print(pd_ts_mean.head())

import statsmodels.tsa.vector_ar.var_model as statsvar
#Vector Autoregression
smmodel = statsvar.VAR(pd_ts_mean); model.select_order(8)

