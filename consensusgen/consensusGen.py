# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:43:09 2024

@author: romain.coulon
"""

import numpy as np
import matplotlib.pyplot as plt

def consensusGen(X,u,*,ng=3,ni=8000):
    """Calculate a reference value using an evolutionary algorithm
    See: Romain Coulon and Steven Judge 2021 Metrologia 58 065007
    DOI 10.1088/1681-7575/ac31c0
    
    Parameters
    ----------
    X : list of floats
        Measurement values.
    u : list of floats
        Standard uncertainties of measurement values.
    ng : int
        Number of generation (Defaul = 3). Set ng=0 for Linear Pool estimation.
    ni : int
        Number of individuals in the whole population (Default = 8000).

    Returns
    -------
    mu : float
        Reference value.
    u_mu : float
        Standard uncertainty of the reference value.
    g0pop : list of floats
        Linear Pool distribution.
    gLpop : list of floats
        EA filtered distribution.
    w : list of floats
        Weigths associated with laboratories.
    """
    
    m=len(X)
    ni=int(ni/m)
    if ng==0: z=0
    else: z=range(ng)   # generation index
    x=range(m)    # group index
    y=range(ni)    # individual index
    Gen=np.asarray(x)+1
    Gen2=np.empty([m,ni])         # genotype of the current generation
    Gen3=np.empty([m,ni])         # genotype of the next generation
    q=np.asmatrix(np.empty([m,ni]))  # phenotype of all the persons at the current generation
    q2=np.asmatrix(np.empty([m,ni])) # phenotype of all the persons at the next generation
    w=np.ones([m,ni])             # weight of the current generation
    KCRV=np.empty(ng)      # mean of the whole population at each generation
    uKCRV=np.empty(ng)     # standard deviation of the whole population at each generation
    # generate the first generation
    for i in x:    q[i]=np.random.normal(X[i],u[i],ni)
    Q2=np.ravel(q) # Suppress the group sepration
    if ng==0:
        KCRV=[np.mean(Q2)]
        uKCRV=[np.std(Q2)/np.sqrt(m)]
        Q=Q2
        G=0
        Wgth=np.ones(m)
    else:
        # attribute a genome to every body (the same within a group)
        for i in x:
            Gen2[i,:]=Gen[i]
            Gen3[i,:]=Gen[i]
        for t in z:  # generation to generation
            q2=np.asmatrix(np.zeros([m,ni]))
            for i in x: # group to group
                for j in y: # person to person
                    if w[i,j]!=0:
                        # search for the nearest phenotype
                        Mat1=abs(q[i,j]-q[:,:]) # deviation of the other compared to ij
                        Mat1[i,j]=float("inf")
                        Idx=np.where(Mat1==Mat1.min())
                        l=Idx[0][0]; c=Idx[1][0]
                        if Gen2[i,j]!=Gen2[l,c]:  # if the genome i'j' difers from ij
                            r=np.random.rand(1)
                            q2[i,j]=r*q[i,j]+(1-r)*q[l,c]         # crossing half the properties of i'j' with ij  
                            Gen3[i,j]=np.sqrt(Gen2[i,j]*Gen2[l,c]) # attribute a new genome
                        else:
                            w[i,j]=0
                            q2[i,j]=q[i,j]
                    else:
                        w[i,j]=0
                        q2[i,j]=q[i,j]                   
            q=q2        # new to current phenotype matrix
            Gen2=Gen3   # new to current genotype matrix  
            Q=np.ravel(q2) # decompartmentalize the groups
            W=np.ravel(w)  # decompartmentalize the weight
            G=np.where(W!=0)
            Wgth=np.empty(m)
            for i in range(m):
                gf=np.where(w[i]!=0)
                Wgth[i]=len(gf[0])/ni
            if ng==0:
                KCRV[t]=np.mean(Q2)
                uKCRV[t]=np.std(Q2)/np.sqrt(m)                
            else:
                KCRV[t]=np.mean(Q[G])
                uKCRV[t]=np.std(Q[G])/np.sqrt(m)
    mu=KCRV[ng-1]
    u_mu=uKCRV[-1]
    g0pop=Q2
    gLpop=Q[G]
    w=Wgth/sum(Wgth)
    return     mu, u_mu, g0pop, gLpop, w

def displayResult(X, u, result, *, lab = False):
    """
    Display the result of the genetic algorithm consensusGen()

    Parameters
    ----------
    X : list of floats
        Measurement values.
    u : list of floats
        Standard uncertainties of measurement values.
    result : list
        Output of consensusGen().
    lab : list, optional
        List of the participants. The default is False.

    Returns
    -------
    None.

    """
    mu, u_mu, g0pop, gLpop, w  = result
    nX = len(X)
    
    print(f"the consensus value is {mu} +/- {u_mu}")
    
    if not lab:
        lab = np.linspace(1,nX,nX)
        labstr = [str(int(x)) for x in lab]
    
    plt.figure("Data")
    plt.clf()
    plt.errorbar(labstr, X, yerr=u, fmt='ok', capsize=3, ecolor='k',label=r"$x_i$")
    plt.plot(lab-1,np.ones(nX)*mu, "-r", label=r"$\hat{\mu}$")
    plt.plot(lab-1,np.ones(nX)*(mu+u_mu), "--r",label=r"$\hat{\mu}+u(\hat{\mu})$")
    plt.plot(lab-1,np.ones(nX)*(mu-u_mu), "--r",label=r"$\hat{\mu}-u(\hat{\mu})$")
    plt.ylabel(r'value', fontsize=12)
    plt.xlabel(r'participant', fontsize=12)
    plt.legend()

    plt.figure("weights")
    plt.clf()
    plt.bar(labstr, w)
    plt.ylabel(r'$w_i$', fontsize=12)
    plt.xlabel(r'participant', fontsize=12)
    plt.legend()
    
    plt.figure("distributions")
    plt.clf()
    plt.hist(g0pop, bins=100, edgecolor='none', density=True, label='linear pooling')
    plt.hist(gLpop, bins=100, edgecolor='none', alpha = 0.7, density=True, label='genetic algorithm')
    plt.ylabel(r'$p(x_i)$', fontsize=12)
    plt.xlabel(r'$x$', fontsize=12)
    plt.legend()


