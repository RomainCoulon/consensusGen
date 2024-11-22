# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:46:53 2024

@author: romain.coulon
"""

from sys import path
path.insert(0, 'G:\Python_modules\BIPM_RI_PyModules')
from DataProcessing import PMM, Unweigthedmean 
from DataProcessingAd import Weightmean, BirgeAdjut, DerSimonianLairdp, CoxProcedureA, CoxProcedureB

import pandas as pd
import consensusgen as cg
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import math


def weighted_median(data, weights):
    """
    Calculate the weighted median of a list of data points with associated weights.

    Parameters:
    data (list of float): The data points.
    weights (list of float): The weights associated with each data point.

    Returns:
    float: The weighted median.
    """
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length.")

    # Sort data and weights together
    sorted_data = sorted(zip(data, weights))

    # Unzip the sorted data and weights
    sorted_data, sorted_weights = zip(*sorted_data)

    # Calculate the cumulative weights
    cumulative_weights = [sum(sorted_weights[:i+1]) for i in range(len(sorted_weights))]

    # Find the median point
    total_weight = sum(sorted_weights)
    median_point = total_weight / 2.0

    # Find the weighted median
    for i in range(len(cumulative_weights)):
        if cumulative_weights[i] >= median_point:
            return sorted_data[i]

    # If the loop completes without returning, return the last data point
    return sorted_data[-1]

def weighted_mad(data, weights):
    """
    Calculate the weighted Median Absolute Deviation (MAD) of a list of data points with associated weights.

    Parameters:
    data (list of float): The data points.
    weights (list of float): The weights associated with each data point.

    Returns:
    float: The weighted MAD.
    """
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length.")

    # Calculate the weighted median
    median = weighted_median(data, weights)

    # Calculate the absolute deviations from the median
    absolute_deviations = [abs(x - median) for x in data]

    # Calculate the weighted median of the absolute deviations
    mad = weighted_median(absolute_deviations, weights)

    return mad


sigma = 1
df = 3
Npoint = 7
outlierPos = 7
normal = True

outlierAmp = np.arange(0,5.5, 0.5)
nMC = 10

burnin = 3
N= Npoint + 2

# Display the DataFrame
if normal:
    x = np.random.normal(0,sigma,N)
    u = np.ones(N)*sigma
else: 
    x = t.rvs(df, size=N)
    if df > 2:
        u = np.ones(N)*np.sqrt(df/(df-2))
    
def t_test(x,u,mu,df):
    t_stat = abs(x-mu)/u    # t-test statistic
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return p_value

tWM = []; tUM = []; tDSL = []; tCPA = []; tCPB = []; tLP = []; tGen = []; tGen2 = []
utWM = []; utUM = []; utDSL = []; utCPA = []; utCPB = []; utLP = []; utGen = []; utGen2 = []

dWM = []; dUM = []; dDSL = []; dCPA = []; dCPB = []; dLP = []; dGen = []; dGen2 = []
udWM = []; udUM = []; udDSL = []; udCPA = []; udCPB = []; udLP = []; udGen = []; udGen2 = []

for j in outlierAmp:
    tWMi = []; tUMi = []; tDSLi = []; tCPAi = []; tCPBi = []; tLPi = []; tGeni = []; tGen2i = []
    LP, Gen, Gen2, UM, WM, DSL, CPA, CPB = [], [], [], [], [], [], [], []
    uLP, uGen, uGen2, uUM, uWM, uDSL, uCPA, uCPB = [], [], [], [], [], [], [], []
    for k in range(nMC):
        if normal:
            x = np.random.normal(0,sigma,N)
            u = np.ones(N)*sigma
        else: 
            x = t.rvs(df, size=N)
            if df > 2:
                u = np.ones(N)*np.sqrt(df/(df-2))
        
        x[outlierPos-1]=j    

        
        result = cg.consensusGen.consensusGen(x, u, ng=1, ni=10000, threshold=0.98)
        # cg.consensusGen.displayResult(x, u, result, lab=sorted_df['Lab'].tolist())
        
        LP.append(result[0][0])
        uLP.append(result[1][0]) 
        Gen.append(result[0][-1])
        uGen.append(result[1][-1])
        
        result = cg.consensusGen.consensusGen(x, u, ng=3, ni=10000, threshold=0.98)
        # cg.consensusGen.displayResult(x, u, result, lab=sorted_df['Lab'].tolist())
         
        Gen2.append(result[0][-1])
        uGen2.append(result[1][-1])
        
        result = Unweigthedmean(x)
        UM.append(result[0])
        uUM.append(result[1])
        
        result = Weightmean(x, u)
        WM.append(result[0])
        uWM.append(result[1])
        
        result = DerSimonianLairdp(x, u)
        DSL.append(result[0])
        uDSL.append(result[1])
        
    
        result = CoxProcedureA(x, u, noFilter=False, k=2.5)
        CPA.append(result[0])
        uCPA.append(result[1])
        
        result = CoxProcedureB(x, u, M=1000)
        CPB.append(result[0])
        uCPB.append(result[1])
        
        
        tWMi.append(t_test(WM[-1],uWM[-1],0,Npoint-1))
        tUMi.append(t_test(UM[-1],uUM[-1],0,Npoint-1)) 
        tDSLi.append(t_test(DSL[-1],uDSL[-1],0,Npoint-1)) 
        tCPAi.append(t_test(CPA[-1],uCPA[-1],0,Npoint-1)) 
        tCPBi.append(t_test(CPB[-1],uCPB[-1],0,Npoint-1)) 
        tLPi.append(t_test(LP[-1],uLP[-1],0,Npoint-1)) 
        tGeni.append(t_test(Gen[-1],uGen[-1],0,Npoint-1))
        tGen2i.append(t_test(Gen2[-1],uGen2[-1],0,Npoint-1))
        
    
    tWM.append(np.mean(tWMi))
    tUM.append(np.mean(tUMi)) 
    tDSL.append(np.mean(tDSLi)) 
    tCPA.append(np.mean(tCPAi)) 
    tCPB.append(np.mean(tCPBi)) 
    tLP.append(np.mean(tLPi)) 
    tGen.append(np.mean(tGeni)) 
    tGen2.append(np.mean(tGen2i))

    utWM.append(np.std(tWMi))
    utUM.append(np.std(tUMi)) 
    utDSL.append(np.std(tDSLi)) 
    utCPA.append(np.std(tCPAi)) 
    utCPB.append(np.std(tCPBi)) 
    utLP.append(np.std(tLPi)) 
    utGen.append(np.std(tGeni)) 
    utGen2.append(np.std(tGen2i))
    
    dWM.append(np.mean(np.abs(WM)))
    dUM.append(np.mean(np.abs(UM))) 
    dDSL.append(np.mean(np.abs(DSL))) 
    dCPA.append(np.mean(np.abs(CPA))) 
    dCPB.append(np.mean(np.abs(CPB))) 
    dLP.append(np.mean(np.abs(LP))) 
    dGen.append(np.mean(np.abs(Gen))) 
    dGen2.append(np.mean(np.abs(Gen2)))

    udWM.append(np.std(np.abs(WM)))
    udUM.append(np.std(np.abs(UM)))
    udDSL.append(np.std(np.abs(DSL)))
    udCPA.append(np.std(np.abs(CPA))) 
    udCPB.append(np.std(np.abs(CPB))) 
    udLP.append(np.std(np.abs(LP))) 
    udGen.append(np.std(np.abs(Gen))) 
    udGen2.append(np.std(np.abs(Gen2))) 


plt.figure("#1")
plt.clf()
plt.title(f"t-test N={Npoint}")
x = outlierAmp
# Plotting each set of error bars
plt.errorbar(x, tWM, yerr=utWM, label='WM', fmt='-')
# plt.errorbar(x, tUM, yerr=utUM, label='tUM', fmt='-')
plt.errorbar(x, tCPA, yerr=utCPA, label='CPA', fmt='-')
plt.errorbar(x, tCPB, yerr=utCPB, label='CPB', fmt='-')
plt.errorbar(x, tDSL, yerr=utDSL, label='DSL', fmt='-')
# plt.errorbar(x, tLP, yerr=utLP, label='tLP', fmt='-')
plt.errorbar(x, tGen, yerr=utGen, label='Gen', fmt='-')
plt.errorbar(x, tGen2, yerr=utGen2, label='Gen2', fmt='-')
plt.ylim([0,1])
plt.ylabel(r'p_value', fontsize=12)
plt.xlabel(r'deviation ($\sigma$)', fontsize=12)
plt.xticks(x, outlierAmp)  # Setting the x-ticks to be the outlierAmp values
plt.legend()
plt.show()

plt.figure("#2")
plt.clf()
plt.title(f"deviation N={Npoint}")
x = outlierAmp
# Plotting each set of error bars
plt.errorbar(x, dWM, yerr=udWM, label='WM', fmt='-')
# plt.errorbar(x, tUM, yerr=utUM, label='tUM', fmt='-')
plt.errorbar(x, dCPA, yerr=udCPA, label='CPA', fmt='-')
plt.errorbar(x, dCPB, yerr=udCPB, label='CPB', fmt='-')
plt.errorbar(x, dDSL, yerr=udDSL, label='DSL', fmt='-')
# plt.errorbar(x, tLP, yerr=utLP, label='tLP', fmt='-')
plt.errorbar(x, dGen, yerr=udGen, label='Gen', fmt='-')
plt.errorbar(x, dGen2, yerr=udGen2, label='Gen2', fmt='-')
# plt.ylim([0,1])
plt.ylabel(r'estimator deviation ($\sigma$)', fontsize=12)
plt.xlabel(r'outlier deviation ($\sigma$)', fontsize=12)
plt.xticks(x, outlierAmp)  # Setting the x-ticks to be the outlierAmp values
plt.legend()
plt.show()