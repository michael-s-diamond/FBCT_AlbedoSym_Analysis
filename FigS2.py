"""
Code to reproduce Figure S2 in Diamond et al. (2024), ESSOAr

Averaging times to converge on climatology (~0 W m-2) for albedo symmetry

Modification history
--------------------
27 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
4 July 2024: Michael Diamond, Tallahassee, FL
    -Added natural experiments
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from cmcrameri import cm as cmc
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from glob import glob
import os
import warnings

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'
dir_fig = '/Users/michaeldiamond/Documents/Projects/Albedo_Sym/FBCT_analysis/'

#Load data
fTr = xr.open_dataset(dir_data+'CERES/EBAFed42/EBAF_gavg_trends.nc')

#
###Calculate averages over different numbers of consecutive months
#

def xavg(N=12):
    """
    Return all samples averaged over N consecutive months
    """
    
    if N == 1: x = fTr['Ra'].sel(reg='NH-SH').values

    else:

        x = fTr['Ra'].sel(reg='NH-SH')[:-N+1].values

        for i in range(1,N-1):
            x += fTr['Ra'].sel(reg='NH-SH')[i:-N+1+i].values

        x += fTr['Ra'].sel(reg='NH-SH')[N-1:].values
        
    return x/N

diff_mean = np.array([np.mean(np.abs(xavg(i))) for i in range(1,121)])
diff_p95 = np.array([np.percentile(np.abs(xavg(i)),95) for i in range(1,121)])
diff_max = np.array([np.max(np.abs(xavg(i))) for i in range(1,121)])
std_err = np.array([np.std(xavg(i))/np.sqrt(234-i) for i in range(1,121)])

mean = np.array([np.mean(xavg(i)) for i in range(1,121)])
p025 = np.array([np.percentile(xavg(i),2.5) for i in range(1,121)])
p25 = np.array([np.percentile(xavg(i),25) for i in range(1,121)])
p75 = np.array([np.percentile(xavg(i),75) for i in range(1,121)])
p975 = np.array([np.percentile(xavg(i),97.5) for i in range(1,121)])
xmax = np.array([np.max(xavg(i)) for i in range(1,121)])
xmin = np.array([np.min(xavg(i)) for i in range(1,121)])


#
###Get natural experiment values
#

dR = {} #Values
dV = {} #Abs values
dT = {} #Length of time
dC = {} #Colors
dM = {} #Markers
dX = {} #x for text
dY = {} #y for text

#Trends
def exp_trends(var,reg,t0,t1):
    
    y = fTr[var].sel(time=slice(t0,t1),reg=reg).values
    x = np.arange(len(y))/12/10
    
    ybar = y.mean()
    xbar = x.mean()
    
    return (np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2)), np.abs(np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2)), len(y)

dR['ARC'], dV['ARC'], dT['ARC'] = exp_trends(var='Ra_sfc',reg='NH',t0=np.datetime64('2002-07-01'),t1=np.datetime64('2012-06-30'))
dC['ARC'], dM['ARC'] = cm.YlOrBr(.33), '^'

dR['CHN'], dV['CHN'], dT['CHN'] = exp_trends(var='Ra_aer',reg='NH',t0=np.datetime64('2010-06-01'),t1=np.datetime64('2019-05-31'))
dC['CHN'],dM['CHN'] = cm.YlOrBr(.67), '^'

#Differences
def exp_diff(var,reg,pre_t0,pre_t1,post_t0,post_t1):
    
    pre = fTr[var].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])).mean().values
    post = fTr[var].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])).mean().values
        
    return (post-pre), np.abs(post-pre), len(fTr[var].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])))

dt0 = {'pre' : {}, 'post' : {}}
dt1 = {'pre' : {}, 'post' : {}}

exp = 'ANT'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-06-01'),np.datetime64('2015-06-01'),np.datetime64('2015-06-01'),np.datetime64('2019-06-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_sfc','SH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.YlOrBr(.33), 'v'

exp = 'IMO'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2017-07-01'),np.datetime64('2020-01-01'),np.datetime64('2020-01-01'),np.datetime64('2022-07-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_cld','NH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.Blues(.5), '^'

exp = 'COV'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-02-01'),np.datetime64('2020-02-01'),np.datetime64('2020-02-01'),np.datetime64('2021-02-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_aer','NH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.YlOrBr(.67), '^'

exp = 'AUS'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-09-01'),np.datetime64('2019-12-01'),np.datetime64('2019-12-01'),np.datetime64('2020-03-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_aer','SH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.YlOrBr(.67), 'v'

exp = 'RAI'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-04-01'),np.datetime64('2019-06-01'),np.datetime64('2019-07-01'),np.datetime64('2019-09-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_aer','NH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.YlOrBr(.67), '^'

exp = 'NAB'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-04-01'),np.datetime64('2011-06-01'),np.datetime64('2011-06-01'),np.datetime64('2011-08-01')
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_aer','NH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dC[exp], dM[exp] = cm.YlOrBr(.67), '^'

#Text
exp = 'ARC'
dX[exp], dY[exp] = dT[exp]-2, dV[exp]
exp = 'CHN'
dX[exp], dY[exp] = dT[exp]-2, dV[exp]
exp = 'ANT'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]
exp = 'IMO'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]
exp = 'COV'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]
exp = 'AUS'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]
exp = 'NAB'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]
exp = 'RAI'
dX[exp], dY[exp] = dT[exp]+2, dV[exp]

#
###Plot
#

plt.figure(figsize=(10,7.5))
plt.clf()

fs = 16

ax1 = plt.subplot(2,1,1)
plt.plot([0,120],[0,0],'k--',lw=1,zorder=11)
plt.plot(np.arange(1,121),mean,c='k',lw=2,label='Mean',zorder=10)
plt.fill_between(np.arange(1,121),p25,p75,facecolor='.5',label='IQR',zorder=3)
plt.fill_between(np.arange(1,121),p025,p975,facecolor='.75',label='95%',zorder=2)
plt.fill_between(np.arange(1,121),xmin,xmax,facecolor='.95',label='Max-min',zorder=1)
plt.plot(np.arange(1,121),xmax,'k',lw=1,ls='dotted')
plt.plot(np.arange(1,121),xmin,'k',lw=1,ls='dotted')

for exp in dM.keys():
    plt.scatter(dT[exp],dR[exp],s=200,color=dC[exp],marker=dM[exp],lw=1,edgecolors='k',zorder=12)
    
plt.legend(frameon=False,fontsize=fs-2,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)

plt.ylim(-4,4)
plt.yticks(fontsize=fs-2)
plt.ylabel(r'$\Delta R^\prime$ ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax1.text(-.125,1,s='(a)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

ax2 = plt.subplot(2,1,2)

plt.plot(np.arange(1,121),diff_mean,c='k',lw=2,label='Mean',zorder=10)
plt.plot(np.arange(1,121),diff_p95,c='k',linestyle='dashed',lw=2,label='95%',zorder=10)
plt.plot(np.arange(1,121),diff_max,c='k',linestyle='dotted',lw=2,label='Max',zorder=10)

for exp in dM.keys():
    plt.scatter(dT[exp],dV[exp],s=200,color=dC[exp],marker=dM[exp],lw=1,edgecolors='k')
    if dX[exp]>dT[exp]: ha = 'left'
    else: ha = 'right'
    if exp == 'NAB': va = 'top'
    else: va = 'center'
    plt.text(dX[exp],dY[exp],exp,fontsize=fs-4,color=dC[exp],ha=ha,va=va)

plt.legend(frameon=False,fontsize=fs-2,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)
plt.xlabel('Number of consecutive months averaged',fontsize=fs)

plt.yscale('log')
plt.ylim(.01,10)
plt.yticks([.01,.1,1,10],[.01,.1,1,10],fontsize=fs-2)
plt.ylabel(r'|$\Delta R^\prime$| ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax2.text(-.125,1,s='(b)',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

plt.savefig(dir_fig+'FigS2.png',dpi=450)













"""
Estimates of overall asymmetry
"""

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fTr.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

#Weighting by month lengths
Dm = np.average(fTr['R'].sel(reg='NH-SH'),weights=Mwts)
r1 = stats.pearsonr(fTr['Ra'].sel(reg='NH-SH')[:-1],fTr['Ra'].sel(reg='NH-SH')[1:])[0]
nu = len(fTr['Ra'].sel(reg='NH-SH'))*(1-r1)/(1+r1)
Ds = np.sqrt(np.average((fTr['Ra'].sel(reg='NH-SH'))**2,weights=Mwts)/(nu-1))

#Not weighting by month lengths
Dm_ = np.mean(fTr['R'].sel(reg='NH-SH'))
r1_ = stats.pearsonr(fTr['Ra'].sel(reg='NH-SH')[:-1],fTr['Ra'].sel(reg='NH-SH')[1:])[0]
nu_ = len(fTr['Ra'].sel(reg='NH-SH'))*(1-r1_)/(1+r1_)
Ds_ = np.sqrt(np.mean((fTr['Ra'].sel(reg='NH-SH'))**2)/(nu_-1))




