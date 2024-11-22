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
import matplotlib.pyplot as plt

radv = ["Na-22","Co-57","Co-60","Ga-67","Ge-68","Sr-85","Y-88","Cd-109","Ag-110m","Sn-113","Ba-133","Cs-134","Ce-139","Gd-153","Sm-153","Tb-161","Lu-177","Tl-201","Ra-223","Ac-225"]
burnin = 2
# ,"Cs-137"
dev_iWM = []
dev_iUM = []
dev_iDSL = []
dev_iCPA = []
dev_iCPB = []
dev_iLP = []
dev_iGen = []
rad_i = []
for rad in radv:
    # columns = ['KCRV', 'Lab', 'Date', 'A', 'u']
    df = pd.read_csv(rad+'.csv',  delimiter=";")
    filtered_df = df[df['KCRV'] == 1]
    # filtered_df = ["?" not in df['Date']]
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d/%m/%Y')
    sorted_df = filtered_df.sort_values(by='Date')
    # Display the DataFrame
    x = np.asarray(sorted_df['A'])
    u = np.asarray(sorted_df['u'])
    
    x = [float(i) for i in x]
    u = [float(i) for i in u]
    
    # df2 = pd.read_csv(rad+'k.csv',  delimiter=";")
    
    # time = np.asarray(sorted_df['Date'])
    # time = sorted_df['Date'].tolist()
    time = np.arange(burnin+1,len(x)+1,1)
    # two-tailed test with alpha = 0.05 (df=n-1)
    critical_t95 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,   # n = 9, df = 8
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,   # n = 19, df = 18
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,   # n = 29, df = 28
        2.045, 2.042, 2.040, 2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024,  # n = 39, df = 38
        2.023, 2.021, 2.020, 2.018, 2.017, 2.015, 2.014, 2.013, 2.012, 2.011,   # n = 49, df = 48
        2.009, 2, 2, 2 , 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ,  # n = 50, df = 49
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ]
    critical_t95 = critical_t95[burnin:len(x)]
    
    def figures(x,u,mu,time,critical_t):
        t = np.sqrt(time)*abs(x-mu)/u    # t-test statistic
        ttest = t>critical_t           # t-test result 95 % 
        dev = abs(x-mu)/x                # relative deviation | accuracy
        sr = u/mu                        # relative std       | precision
        fom = sum(1/time)**-1*sum(dev/time) # figure of merit
        return t, ttest, dev, sr, fom
    
    
    LP, Gen, UM, WM, DSL, CPA, CPB = [], [], [], [], [], [], []
    uLP, uGen, uUM, uWM, uDSL, uCPA, uCPB = [], [], [], [], [], [], []

    for i in range(len(x)+1):
        if i>burnin:
            result = cg.consensusGen.consensusGen(x[:i], u[:i], ng=2, ni=10000, threshold=0.98)
            # cg.consensusGen.displayResult(x, u, result, lab=sorted_df['Lab'].tolist())
            LP.append(result[0][0])
            uLP.append(result[1][0])    
            Gen.append(result[0][-1])
            uGen.append(result[1][-1])
            
            result = Unweigthedmean(x[:i])
            UM.append(result[0])
            uUM.append(result[1])
            
            result = Weightmean(x[:i], u[:i])
            WM.append(result[0])
            uWM.append(result[1])
            
            result = DerSimonianLairdp(x[:i], u[:i])
            DSL.append(result[0])
            uDSL.append(result[1])
            
            if rad == "Cs-137":
                result = CoxProcedureA(x[:i], u[:i], noFilter=True)
            else:
                result = CoxProcedureA(x[:i], u[:i], noFilter=False, k=2.5)
            CPA.append(result[0])
            uCPA.append(result[1])
            
            result = CoxProcedureB(x[:i], u[:i], M=1000)
            CPB.append(result[0])
            uCPB.append(result[1])
        
    nPoint = 5
    endPoint = 10
    if len(LP) >= endPoint:
        # mu=np.mean([LP[-1],Gen[-1],UM[-1],WM[-1]]) # hypothesized true mean
        mu=np.mean([LP[endPoint-1],Gen[endPoint-1],UM[endPoint-1],WM[endPoint-1]]) # hypothesized true mean
        
        t_WM, ttest_WM, dev_WM, sr_WM, fom_WM = figures(WM, uWM, mu, time, critical_t95)
        t_UM, ttest_UM, dev_UM, sr_UM, fom_UM = figures(UM, uUM, mu, time, critical_t95)
        
        t_DSL, ttest_DSL, dev_DSL, sr_DSL, fom_DSL = figures(DSL, uDSL, mu, time, critical_t95)
        t_CPA, ttest_CPA, dev_CPA, sr_CPA, fom_CPA = figures(CPA, uCPA, mu, time, critical_t95)
        t_CPB, ttest_CPB, dev_CPB, sr_CPB, fom_CPB = figures(CPB, uCPB, mu, time, critical_t95)
        
        t_LP, ttest_LP, dev_LP, sr_LP, fom_LP = figures(LP, uLP, mu, time, critical_t95)
        t_Gen, ttest_Gen, dev_Gen, sr_Gen, fom_Gen = figures(Gen, uGen, mu, time, critical_t95)
        
        dev_iWM.append(dev_WM[3])
        dev_iUM.append(dev_UM[3])
        dev_iDSL.append(dev_DSL[3])
        dev_iCPA.append(dev_CPA[3])
        dev_iCPB.append(dev_CPB[3])
        dev_iLP.append(dev_LP[3])
        dev_iGen.append(dev_Gen[3])
        rad_i.append(rad)
        
        meth = ['WM', 'UM', "DSL", "CPA", "CPB", 'LP', 'Gen']
        fom = np.array([fom_WM, fom_UM, fom_DSL, fom_CPA, fom_CPB, fom_LP, fom_Gen])
        ttest = np.array([sum(ttest_WM),sum(ttest_UM),sum(ttest_DSL),sum(ttest_CPA),sum(ttest_CPB),sum(ttest_LP),sum(ttest_Gen)])
        
        # plt.figure("#1")
        # plt.clf()
        # plt.title("estimators")
        # plt.errorbar(time[burnin:], UM, yerr=uUM, fmt='.-g', capsize=3, ecolor='g', label=r"$y$ (UM)")
        # plt.errorbar(time[burnin:], WM, yerr=uWM, fmt='.-b', capsize=3, ecolor='b', label=r"$y$ (WM)")
        # plt.errorbar(time[burnin:], DSL, yerr=uDSL, fmt='.-m', capsize=3, ecolor='m', label=r"$y$ (DSL)")
        # plt.errorbar(time[burnin:], LP, yerr=uLP, fmt='.-k', capsize=3, ecolor='k', label=r"$y$ (LP)")
        # plt.errorbar(time[burnin:], Gen, yerr=uGen, fmt='.-r', capsize=3, ecolor='r', label=r"$y$ (Gen)")
        # plt.ylabel(r'Value', fontsize=12)
        # plt.xlabel(r'Date', fontsize=12)
        # plt.legend()
        
        plt.figure(f"#2 {rad}")
        plt.clf()
        
        plt.title(f"fom {rad}")
        plt.bar(meth, fom, label=r"rad")
        plt.ylabel(r'fom', fontsize=12)
        plt.xlabel(r'method', fontsize=12)
        plt.legend(loc='upper left')
        
        ax2 = plt.gca().twinx()
        ax2.plot(meth, ttest, color='red', marker='o', label=r"t-test")
        ax2.set_ylabel(r't-test', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
        
        # plt.figure("#2")
        # plt.clf()
        # plt.title("FOM")
        # plt.plot(time[burnin:], FOM, '.-g', label=r"$y$ (UM)")
        # # plt.errorbar(time[burnin:], WM, yerr=uUM, fmt='.-b', capsize=3, ecolor='g', label=r"$y$ (WM)")
        # # plt.errorbar(time[burnin:], LP, yerr=uLP, fmt='.-k', capsize=3, ecolor='k', label=r"$y$ (LP)")
        # # plt.errorbar(time[burnin:], Gen, yerr=uGen, fmt='.-r', capsize=3, ecolor='r', label=r"$y$ (Gen)")
        # plt.ylabel(r'Value', fontsize=12)
        # plt.xlabel(r'Date', fontsize=12)
        # plt.legend()

plt.figure("#2")
plt.clf()
plt.title("rate")
plt.plot(rad_i, dev_iWM, label=r"$r$ (WM)")
plt.plot(rad_i, dev_iUM, label=r"$r$ (UM)")
plt.plot(rad_i, dev_iDSL, label=r"$r$ (DSL)")
plt.plot(rad_i, dev_iCPA, label=r"$r$ (CPA)") 
plt.plot(rad_i, dev_iCPB, label=r"$r$ (CPB)")
plt.plot(rad_i, dev_iLP, label=r"$r$ (LP)")
plt.plot(rad_i, dev_iGen, label=r"$r$ (Gen)")
plt.ylabel(r'$d_r$', fontsize=12)
plt.xlabel(r'radionuclide', fontsize=12)
plt.legend()

print('results')
print('WM', np.mean(dev_iWM))
print('UM', np.mean(dev_iUM))
print('DSL', np.mean(dev_iDSL))
print('CPA', np.mean(dev_iCPA))
print('CPB', np.mean(dev_iCPB))
print('LP', np.mean(dev_iLP))
print('Gen', np.mean(dev_iGen))

