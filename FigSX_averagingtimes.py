"""
Code to reproduce Figure SX in Diamond et al. (2023), GRL

Averaging times to converge on climatology (~0 W m-2) for albedo symmetry

Modification history
--------------------
27 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
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
fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')

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
diff_max = np.array([np.max(np.abs(xavg(i))) for i in range(1,121)])
std_err = np.array([np.std(xavg(i))/np.sqrt(234-i) for i in range(1,121)])

mean = np.array([np.mean(xavg(i)) for i in range(1,121)])
p05 = np.array([np.percentile(xavg(i),5) for i in range(1,121)])
p25 = np.array([np.percentile(xavg(i),25) for i in range(1,121)])
p75 = np.array([np.percentile(xavg(i),75) for i in range(1,121)])
p95 = np.array([np.percentile(xavg(i),95) for i in range(1,121)])
xmax = np.array([np.max(xavg(i)) for i in range(1,121)])
xmin = np.array([np.min(xavg(i)) for i in range(1,121)])

#
###Plot
#

plt.figure(figsize=(10,7.5))
plt.clf()

fs = 16

ax1 = plt.subplot(2,1,1)
plt.plot([0,120],[0,0],'k--',lw=1,zorder=11)
plt.plot(np.arange(1,121),mean,c='k',lw=2,label='Mean',zorder=10)
plt.fill_between(np.arange(1,121),p25,p75,facecolor='.5',label='25th-75th',zorder=3)
plt.fill_between(np.arange(1,121),p05,p95,facecolor='.75',label='5th-95th',zorder=2)
plt.fill_between(np.arange(1,121),xmin,xmax,facecolor='.95',label='Max-min',zorder=1)
plt.plot(np.arange(1,121),xmax,'k',lw=1,ls='dotted')
plt.plot(np.arange(1,121),xmin,'k',lw=1,ls='dotted')

plt.legend(frameon=False,fontsize=fs-2,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)

plt.ylim(-4,4)
plt.yticks(fontsize=fs-2)
plt.ylabel(r'$\Delta R^\prime$ ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax1.text(-.125,1,s='a',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

ax2 = plt.subplot(2,1,2)

plt.plot(np.arange(1,121),diff_mean,c='k',lw=2,label='Mean',zorder=10)
plt.plot(np.arange(1,121),diff_max,'k--',lw=2,label='Max',zorder=10)

plt.legend(frameon=False,fontsize=fs-2,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)
plt.xlabel('Number of consecutive months averaged',fontsize=fs)

plt.yscale('log')
plt.ylim(.05,5)
plt.yticks([.1,1],[.1,1],fontsize=fs-2)
plt.ylabel(r'|$\Delta R^\prime$| ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax2.text(-.125,1,s='b',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

plt.savefig(dir_fig+'FigSX_averagingtimes.png',dpi=150)






















