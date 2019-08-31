# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:48:15 2019

@author: Matt
"""

import numpy as np
import pandas as pd
from scipy import stats
#
#correct = np.loadtxt("distcorrect1000.csv")
#
#incorrect = np.loadtxt("dist2wrong1000.csv")
#
#incorrect2=np.loadtxt("distwrong10000.csv")
#
#unknown=np.loadtxt("dist.csv")
#

K=pd.DataFrame()


#print(stats.ks_2samp(correct,incorrect))
var=["correct","incorrect","incorrect2","unknown"]

for a in var:
    for b in var:
        exec("A="+a)
        exec("B="+b)
        print(a,b,stats.ks_2samp(A,B)[1])

print(K)
    
    

b= "correct"

