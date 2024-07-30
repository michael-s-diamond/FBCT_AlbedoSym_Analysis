"""
Code to reproduce Figure S6 in Diamond et al. (2024), ESSOAr


Times series for "natural experiments"

Modification history
--------------------
4 July 2024: Michael Diamond, Tallahassee, FL
    -Created
"""

#Import libraries
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from glob import glob
import os

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'
dir_fig = '/Users/michaeldiamond/Documents/Projects/Albedo_Sym/FBCT_analysis/'

#Load data
fbct = xr.open_mfdataset(glob(dir_data+'CERES/FluxByCldType/FBCT_decomp*nc'))
fbct['Ra_low'] = fbct['Ra_Cu']+fbct['Ra_Sc']+fbct['Ra_St']
fbct['Ra_mid'] = fbct['Ra_Ac']+fbct['Ra_As']+fbct['Ra_Ns']
fbct['Ra_high'] = fbct['Ra_Ci']+fbct['Ra_Cs']+fbct['Ra_Cb']

fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']
fTr['Ra_low'] = fTr['Ra_cld'][:3].sum(axis=0)
fTr['Ra_mid'] = fTr['Ra_cld'][3:6].sum(axis=0)
fTr['Ra_high'] = fTr['Ra_cld'][6:].sum(axis=0)

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fTr.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

"""
Time series
"""

dColors = {'Ra_sfc' : cm.YlOrBr(.33),'Ra_aer' : cm.YlOrBr(.67), 'Ra_low' : cm.Blues(.5)}
dLab = {'Ra_sfc' : r'$R^\prime_\mathrm{sfc}$','Ra_aer' : r'$R^\prime_\mathrm{aer}$', 'Ra_low' : r'$R^\prime_\mathrm{low}$'}
fs = 16

dt0 = {'pre' : {}, 'post' : {}}
dt1 = {'pre' : {}, 'post' : {}}
dVar = {}
dReg = {}
dY = {} #y lim


#
###"Trend" natural experiments
#

def calc_trend(exp):
    
    y = fTr[dVar[exp]].sel(reg=dReg[exp],time=slice(dt0['post'][exp], dt1['post'][exp])).values
    x = np.arange(len(y))/12/10
    
    ybar = y.mean()
    xbar = x.mean()
    
    beta_hat = np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar
    
    yhat = alpha_hat+beta_hat*x
    e = y-yhat
    
    xx = e[1:]
    yy = e[:-1]
    xm = xx.mean()
    ym = yy.mean()
    
    r1e_num = np.sum((xx-xm)*(yy-ym))
    r1e_denom = np.sum((xx-xm)**2)*np.sum((yy-ym)**2)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = len(y)*(1-r1e)/(1+r1e)
    if nue>len(y): nue = len(y)
    
    sbh = np.sqrt(np.sum(e**2)/(nue-2)/np.sum((x-xbar)**2))
    sah = sbh*np.sqrt(np.sum(x**2)/nue)
    
    t = stats.t.ppf(.975,nue-2)
    
    c1 = np.sum(e**2)/(nue-2)
    c2 = 1/len(x) + (x-x.mean())**2/np.sum((x-x.mean())**2)
    
    return x, alpha_hat, beta_hat, t*np.sqrt(c1*c2)

def plot_trend(exp):

    var = dVar[exp]
    reg = dReg[exp]
    ymax = dY[exp]

    x, a, b, c = calc_trend(exp)
    
    t = fTr.time.sel(time=slice(dt0['post'][exp], dt1['post'][exp]))

    plt.plot(t,a+b*x,c=cm.Greens(.67),lw=3,solid_capstyle='butt')

    plt.plot(2*[t[0].values],[-2,2],'k',ls='dotted',lw=.5)
    plt.plot(2*[t[-1].values],[-2,2],'k',ls='dotted',lw=.5)

    plt.fill_between(t.values,a+b*x+c,a+b*x-c,facecolor=cm.Greens(.25),zorder=0)
    plt.scatter(fTr.time[6::12].values[:-1],fTr[var].sel(reg=reg).rolling(time=12,center=True).mean()[::12][1:],s=25,color=dColors[var],lw=.5,edgecolors='k')
    plt.plot(fTr.time,fTr[var].sel(reg=reg).rolling(time=4*12,center=True).mean(),c='.25',lw=2)
    plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=.5)

    plt.xlim(fTr.time[0].values,fTr.time[-1].values)
    plt.ylim(-ymax,ymax)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.ylabel('%s %s' % (dReg[exp],dLab[var]),fontsize=fs)
    
    plt.title(exp,fontsize=fs,loc='left',fontweight='bold')

exp = 'ARC'
dVar[exp] = 'Ra_sfc'
dReg[exp] = 'NH'
dt0['post'][exp], dt1['post'][exp] = np.datetime64('2002-07-01'),np.datetime64('2012-06-30')
dY[exp] = 1

exp = 'CHN'
dVar[exp] = 'Ra_aer'
dReg[exp] = 'NH'
dt0['post'][exp], dt1['post'][exp] = np.datetime64('2010-06-01'),np.datetime64('2019-05-31')
dY[exp] = .8

#
###"Difference" natural exeriments
#

def plot_diff(exp):
    var = dVar[exp]
    reg = dReg[exp]
    ymax = dY[exp]

    pre = fTr[var].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])).mean().values
    post = fTr[var].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])).mean().values

    nt = len(fTr[var].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])))

    plt.scatter(fTr.time[6::12].values[:-1],fTr[var].sel(reg=reg).rolling(time=12,center=True).mean()[::12][1:],color=dColors[var],s=25,lw=.5,edgecolors='k')
    plt.plot(fTr.time,fTr[var].sel(reg=reg).rolling(time=nt,center=True).mean(),c='.25',lw=2)
    plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=.5)

    plt.plot([dt0['pre'][exp],dt1['pre'][exp]],2*[pre],c=cm.Greens(.33),lw=3,solid_capstyle='butt')
    plt.plot([dt0['post'][exp],dt1['post'][exp]],2*[post],c=cm.Greens(.67),lw=3,solid_capstyle='butt')

    plt.plot(2*[dt0['pre'][exp]],[-2,2],'k',ls='dotted',lw=.5)
    plt.plot(2*[dt1['pre'][exp]],[-2,2],'k',ls='dotted',lw=.5)
    if dt0['post'][exp] != dt1['pre'][exp]: plt.plot(2*[dt0['post'][exp]],[-2,2],'k',ls='dotted',lw=.5)
    plt.plot(2*[dt1['post'][exp]],[-2,2],'k',ls='dotted',lw=.5)

    plt.fill_between([dt0['pre'][exp],dt1['pre'][exp]],[-2,-2],[2,2],facecolor=cm.Greens(.01),zorder=0)
    plt.fill_between([dt0['post'][exp],dt1['post'][exp]],[-2,-2],[2,2],facecolor=cm.Greens(.25),zorder=0)

    plt.ylim(-ymax,ymax)
    plt.xlim(fTr.time[0].values,fTr.time[-1].values)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.ylabel('%s %s' % (dReg[exp],dLab[var]),fontsize=fs)
    
    plt.title(exp,fontsize=fs,loc='left',fontweight='bold')
    
    


#ANT
exp = 'ANT'
dVar[exp] = 'Ra_sfc'
dReg[exp] = 'SH'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-06-01'),np.datetime64('2015-06-01'),np.datetime64('2015-06-01'),np.datetime64('2019-06-01')
dY[exp] = 0.5

#IMO 2020
exp = 'IMO'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2017-07-01'),np.datetime64('2020-01-01'),np.datetime64('2020-01-01'),np.datetime64('2022-07-01')
dVar[exp] = 'Ra_low'
dReg[exp] = 'NH'
dY[exp] = 0.5

#COVID-19
exp = 'COV'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-02-01'),np.datetime64('2020-02-01'),np.datetime64('2020-02-01'),np.datetime64('2021-02-01')
dVar[exp] = 'Ra_aer'
dReg[exp] = 'NH'
dY[exp] = .75

exp = 'COV-low'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-02-01'),np.datetime64('2020-02-01'),np.datetime64('2020-02-01'),np.datetime64('2021-02-01')
dVar[exp] = 'Ra_low'
dReg[exp] = 'NH'
dY[exp] = 0.75

#Australian wildfires
exp = 'AUS'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-09-01'),np.datetime64('2019-12-01'),np.datetime64('2019-12-01'),np.datetime64('2020-03-01')
dVar[exp] = 'Ra_aer'
dReg[exp] = 'SH'
dY[exp] = 1.5

exp = 'AUS-low'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-09-01'),np.datetime64('2019-12-01'),np.datetime64('2019-12-01'),np.datetime64('2020-03-01')
dVar[exp] = 'Ra_low'
dReg[exp] = 'SH'
dY[exp] = 1.5

#Nabro
exp = 'NAB'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-04-01'),np.datetime64('2011-06-01'),np.datetime64('2011-06-01'),np.datetime64('2011-08-01')
dVar[exp] = 'Ra_aer'
dReg[exp] = 'NH'
dY[exp] = 2
  
#Raikoke
exp = 'RAI'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-04-01'),np.datetime64('2019-06-01'),np.datetime64('2019-07-01'),np.datetime64('2019-09-01')
dVar[exp] = 'Ra_aer'
dReg[exp] = 'NH'
dY[exp] = 2


    
"""
Make figure
""" 

#Aesthetics and labels
lab = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

#
###Plot
#
plt.figure(figsize=(12,9))
plt.clf()

fs = 12

n = 0
for exp in dVar.keys():
    
    ax1 = plt.subplot(5,2,n+1)
    
    if exp in ['ARC','CHN']: plot_trend(exp)
    else: plot_diff(exp)
        
    #ax1.text(-.075,.45,s=exp,transform = ax1.transAxes,fontsize=fs,fontweight='bold',ha='right')
    
    ax1.text(-.15,1,s='(%s)' % lab[n],transform = ax1.transAxes,fontsize=fs-2,fontweight='bold')
    
    n += 1

plt.tight_layout()

plt.savefig(dir_fig+'FigS6.png',dpi=450)

