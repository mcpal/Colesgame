# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:30:25 2018

@author: Matt
"""

"""
1) Coles game. Collect all 30! How many tries to collect them all?
2) partially completed Coles game. Got n, how many tries to get the rest?
"""

import pandas as pd
import numpy as np

def dist_summary(dist, names = 'dist_name'):
    import pandas as pd
    import matplotlib.pyplot as plt
    ser = pd.Series(dist)
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.gca()
    ser.hist(ax = ax, bins = int(max(dist)-min(dist)))
    ax.set_title('Frequency distribution of ' + names)
    ax.set_ylabel('Frequency')
    plt.show()
    print(ser.describe())
    return(ser.describe())

def scipy_summary(dist):
    from scipy.stats import kurtosis, skew
    print("mean : ", np.mean(dist))
    print("var  : ", np.var(dist))
    print("skew : ",skew(dist))
    print("kurt : ",kurtosis(dist))
#    mvsk    
    
def distgen(numtoys=30,numcustomers=1000):
    #initialise toy array and distribution list.
    toy=pd.DataFrame(np.zeros((numtoys,numcustomers),dtype=np.int32))
    toy.index=range(1,numtoys+1)
    dist=[]
    #all customers play at the beginning
    remcustomers=numcustomers

    #sequential visits to Coles
    while remcustomers >0 :
        toy=statsgen(remcustomers,toy,numtoys)
        for a in range(0,remcustomers): #for every column in toy (every player/customer)
            if a in toy.columns and min(toy.loc[:,a])>0: #this means got every toy at least once - i.e. bingo!
                dist.append(sum(toy.loc[:,a])) #put the number of visits (= number of toys) in the distribution list
                del toy[a] # remove this column from the dataframe, they're done.
                remcustomers-=1 # we can reduce the number of randoms to generate next time
                toy.columns=range(0,remcustomers) # redo the column index so statsgen doesn't get confused
    #            print(dist)
                print(remcustomers) #give the user some feedback about how many are left in the simulation
    return dist;
    

def statsgen(numcustomers,toy,numtoys):
#    global toy
    
    import numpy.random as nr
    unif = [int(x) for x in nr.uniform(low=1, high=numtoys+1,size = numcustomers)]
    
    for a in range(0,numcustomers):
        toy.loc[unif[a],a]+=1
    return toy;

def savedist(dist,filname="dist.csv"):
    np.savetxt("dist.csv",dist)

def loaddist():
    dist=np.loadtxt("dist.csv")
    data=pd.Series(dist)
    disti=np.array(dist,dtype=int)
    bc=np.bincount(disti)

def loadfits():
    fish=np.load("fits.npy")
    fits=fish.item()
    dfits=pd.DataFrame.from_dict([fits])
    
def getbestfive():
    dfits.dropna(axis=1,inplace=True)
    dfits.sort_values(0,axis=1,inplace=True)
    dfits.iloc[:,0:4]
    dfits.iloc[0,0:4]/dfits.iloc[0,0]
    dfits.iloc[0,0:16]/dfits.iloc[0,0]


def run():
    dist_summary(dist, 'number of visits to get all')