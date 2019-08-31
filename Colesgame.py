# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:30:25 2018

@author: Matt

https://stackoverflow.com/questions/45483890/how-to-correctly-use-scipys-skew-and-kurtosis-functions#45484287


1) Coles game. Collect all 30! How many tries to collect them all?
2) partially completed Coles game. Got n, how many tries to get the rest?
"""
import pandas as pd
import numpy as np


def dist_summary(dist, names = 'dist_name'):
    import matplotlib.pyplot as plt
    ser = pd.Series(dist)
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.gca()
    #bing= max(int(max(dist)-min(dist)),1)
    ser.hist(ax = ax)#, bins=bing)
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
  
def statsgen(toy,n,numcustomers):
    import numpy.random as nr
    unif = [int(x) for x in nr.uniform(low=1, high=n+1,size = numcustomers)]
    
    for a in range(0,numcustomers):
        toy.loc[unif[a],a]+=1
    return toy;


def makedist(n=30,numcustomers=1000,alreadyhave=28):
    #initialise
    if alreadyhave >0:
        toy=pd.DataFrame(np.ones((alreadyhave,numcustomers),dtype=np.int32)) 
        toy=toy.append(pd.DataFrame(np.zeros((n-alreadyhave,numcustomers),dtype=np.int32)),ignore_index=True)
    else:
        toy=pd.DataFrame(np.zeros((n,numcustomers),dtype=np.int32))
        
    toy.index=range(1,n+1)
    dist=[]
    
    remcustomers=numcustomers

    #sequential visits to Coles
    while remcustomers >0 :
        #Check whether anyone's finished
        for a in range(0,remcustomers): #for every column in toy (every player/customer)
            if a in toy.columns and min(toy[a])>0: #this means got every toy at least once - i.e. bingo!
    #            print(a,remcustomers)
                dist.append(sum(toy.loc[:,a])-alreadyhave) #put the number of visits (= number of toys-toys already have) in the distribution list
                del toy[a] # remove this column from the dataframe, they're done.
                remcustomers-=1 # we can reduce the number of randoms to generate next time
    #            print(dist)
    #            print(remcustomers)
        toy.columns=range(0,remcustomers) # redo the column index so statsgen doesn't get confused
    #    print("loop done" + str(remcustomers))
        # Give some progress info to user
        print(remcustomers)
        
        #Grab some random numbers.
        if remcustomers>0: statsgen(toy,n,remcustomers)

    return dist;


def makedistdist(): #dist by alreadyhave
    ahdist={}
    for k in range(0,31):
        ahdist[k]=pd.Series(makedist(n=30,numcustomers=100,alreadyhave=k)).describe()[1]
    plt.plot(ahdist.keys(),ahdist.values())
    return ahdist;

def makendist(mi=1,ma=31,ahdist={}):
#    ahdist={}
    for k in range(mi,ma):
        ahdist[10*k]=pd.Series(makedist(n=10*k,numcustomers=100,alreadyhave=0)).describe()[1]
    plt.plot(ahdist.keys(),ahdist.values())
    return ahdist;

def exacts(mi=1,ma=180):
    y={}
    y[1]=1
    for x in range(2,ma):
        y[x]= x*y[x-1]/(x-1)+1
    return y;




def controls():
    dist_summary(dist, 'number of visits to get all')
#    np.savetxt("dist.csv",dist)

def loaddist(file="dist.csv"):
    dist=np.loadtxt(file)
    data=pd.Series(dist)
    disti=np.array(dist,dtype=int)
    bc=np.bincount(disti)

def fitsload(file="fits.npy"):
    fish=np.load("fits.npy")
    fits=fish.item()
    dfits=pd.DataFrame.from_dict([fits])
    
    
def getbestfive():
    dfits.dropna(axis=1,inplace=True)
    dfits.sort_values(0,axis=1,inplace=True)
    dfits.iloc[:,0:4]
    dfits.iloc[0,0:4]/dfits.iloc[0,0]
    dfits.iloc[0,0:16]/dfits.iloc[0,0]

'''
https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy


plt.plot(list(yha.keys()),list(yha.values()))
Out[113]: [<matplotlib.lines.Line2D at 0x258f1bafd68>]

plt.plot(list(yha.keys()),list(yha.values()))
Out[114]: [<matplotlib.lines.Line2D at 0x258f1bafc88>]

plt.plot(ahdist.keys(),ahdist.values())
Out[115]: [<matplotlib.lines.Line2D at 0x258f1c09f60>]

plt.plot(n3dist.keys(),n3dist.values())
Out[116]: [<matplotlib.lines.Line2D at 0x258f1c19fd0>]

yha=exacts()


linregress(list(n3dist.keys()),list(n3dist.values()))
fit=np.polyfit(list(n3dist.keys()),list(n3dist.values()),2)
fit
fitlin=__
fitlin
__
_
fitlin=linregress(list(n3dist.keys()),list(n3dist.values()))
fit1=np.polyfit(list(n3dist.keys()),list(n3dist.values()),1)
'''