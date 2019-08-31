# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 22:09:43 2018

@author: tmthydvnprt and Josef
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
and
https://stackoverflow.com/questions/31359017/how-to-get-error-estimates-for-fit-parameters-in-scipy-stats-gamma-fit

https://stackoverflow.com/questions/45483890/how-to-correctly-use-scipys-skew-and-kurtosis-functions#45484287
"""

#%matplotlib inline



import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')



def fits_error_estimates(distribution,data,params):
    from statsmodels.base.model import GenericLikelihoodModel
        
    
    param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
    
    
    
    print(str(distribution.name) + '\n' + str(param_names) + '\n' + str(params) + '\n')
      
    class Modeler(GenericLikelihoodModel):
    
        nparams = len(params)
    
        def loglike(self, params):
            return distribution.logpdf(self.endog, *params).sum()
    
    
    res = Modeler(data).fit(start_params=params)
    res(extra_params_names=param_names)
    res.df_model = len(params)
    res.df_resid = len(data) - len(params)
    res.params
    print(res.summary())
#    print(res.t_test())


def distributionlist(distro=0):
    if distro==1:
        DISTRIBUTIONS = [st.alpha,st.betaprime,st.exponweib,st.f,st.fatiguelife,st.genlogistic,st.genextreme,#st.gengamma,
            st.gumbel_r,st.invgamma,st.invgauss,st.johnsonsb,st.johnsonsu,st.lognorm,st.nct,st.recipinvgauss,
        ]
    elif distro ==2:
        DISTRIBUTIONS=[st.maxwell]
    elif distro == "discrete":
        DISTRIBUTIONS = [st.bernoulli,st.binom,st.boltzmann,st.dlaplace,st.geom,st.hypergeom,st.logser,
                         st.nbinom,st.planck,st.poisson,st.randint,st.skellam,st.zipf,st.yulesimon]
    elif distro == 0:
        DISTRIBUTIONS = [        
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]
    else:
        DISTRIBUTIONS=distro
        
    return DISTRIBUTIONS;



def fit_distribution(data, bins=200,distro=0,discrete=False):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    if discrete==True:
        x = range(min(data),max(data)+1)
#        dist=pd.Series(dist)
#    disti=np.array(dist,dtype=int)
#    bc=np.bincount(disti)
        
        y = data
    else:
        y, x = np.histogram(data, bins=bins, density=True)
    
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS=distributionlist(distro)
        
    fits=[]
    
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                fits.append([distribution.name,sse,pd.Series(pdf, x),params])
#                print(distribution,sse)
                   
        except Exception:
            pass
    
    return fits;

def best_fits(fits,datatype="list",num=5):
    if datatype=="list":
        dfits=pd.DataFrame(fits)
    elif datatype=="dict":
        dfits=pd.DataFrame.from_dict([fits])
    
    dfits.columns=["Dist","SSE","pdf","params"]
    dfits.dropna(axis=0,inplace=True)
    #note used to be axis=1
    dfits.sort_values(by=['SSE'],axis=0,inplace=True)
    #note, required (0,axis=1) previously
    bfits=dfits.iloc[0:num,:]
#    note required: bfits=dfits.iloc[:,0:num] previously
    bfits.index=range(0,len(bfits))
    return bfits;
    
  

def make_pdf(dist, params, size=10000,tailschop=0.001):
    """Generate distribution's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(tailschop, *arg, loc=loc, scale=scale) if arg else dist.ppf(tailschop, loc=loc, scale=scale)
    end = dist.ppf(1-tailschop, *arg, loc=loc, scale=scale) if arg else dist.ppf(1-tailschop, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def plotdataandfitpdf(data,pdf,name='',params='',SSE=''):
# Plot all of the fits, + data.    
    
    #def plotdists(data,bins=50):    
        # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    #    ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
        # Save plot limits
    dataYLim = ax.get_ylim()

        # if axis pass in add to plot
#    try:
#        if ax:
#            pd.Series(pdf, x).plot(ax=ax)
#        end
#    except Exception:
#        pass

    pdf.plot(ax=ax)

    # Update plots
    ax.set_ylim(dataYLim)
    param_names = (getattr(st,name).shapes + ', loc, scale').split(', ') if getattr(st,name).shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, params)])
    dist_str = '{}({})'.format(name, param_str)


#    ax.set_title(u'.\n Data and ' + name + ' fit with ' + str(params) + '\n SSE=' + str(SSE))
    ax.set_title(u' Data and fit with \n' + dist_str + '\n SSE=' + str(SSE))
    ax.set_xlabel(u'x')
    ax.set_ylabel('Frequency')


def plotdataandfitparams(data,best_fit_name,best_fit_params):
    best_dist = getattr(st, best_fit_name)
    
    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    
    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    
    ax.set_title(u' Best fit distribution \n' + dist_str)
    ax.set_xlabel(u'x')
    ax.set_ylabel('y')
    

    
def plotfits(data,bfits):

    for i in range(0,len(bfits)):
        plotdataandfitpdf(data,bfits.iloc[i,2],bfits.iloc[i,0],bfits.iloc[i,3],bfits.iloc[i,1])

        
        
        
def runit(data):
    fits=fit_distribution(data)
    bfits=best_fits(fits)
    plotfits(data,bfits)
    return bfits;


def check():
    for i in bfits.index:
        fits_error_estimates(getattr(st,bfits.Dist[i]),data,list(bfits.params[i]))