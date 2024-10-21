"""
Code to reproduce Figure S5 in Diamond et al. (2024), GRL

Natural experiments analysis

Modification history
--------------------
21 September 2024: Michael Diamond, Tallahassee, FL
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

"""
Figure S5: Zonal trends
"""

dZt = {} #Trends
dZe = {} #Standard errors

for var in ['Ra','Ra_sfc','Ra_aer','Ra_low','Ra_mid','Ra_high']:
    y = fbct[var].mean(axis=-1)
    ybar = y.mean(axis=0)
    x = np.arange(len(fbct.time))
    xbar = x.mean()
    
    #Calculate slope and intercept
    beta_hat = np.sum((x-xbar)[:,np.newaxis]*(y-ybar),axis=0)/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar
    
    yhat = alpha_hat.values[np.newaxis,:]+beta_hat.values[np.newaxis,:]*x[:,np.newaxis]
    e = y-yhat
    
    xx = e[1:,:].values
    yy = e[:-1,:].values
    xm = xx.mean(axis=0)[np.newaxis,:]*np.ones(xx.shape)
    ym = yy.mean(axis=0)[np.newaxis,:]*np.ones(yy.shape)
    
    r1e_num = np.sum((xx-xm)*(yy-ym),axis=0)
    r1e_denom = np.sum((xx-xm)**2,axis=0)*np.sum((yy-ym)**2,axis=0)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = len(x)*(1-r1e)/(1+r1e)
    nue[nue>len(x)] = len(x) #Can't have DOF larger than nt
    
    sbh = np.sqrt(np.sum(e**2,axis=0)/(nue-2)/np.sum((x-xbar)**2))
    
    dZt[var] = 120*10*beta_hat
    dZe[var] = 120*10*sbh
    

#
###Plot
#
ylab = [r'$R^\prime$',r'$R^\prime_{\mathrm{sfc}}$',r'$R^\prime_{\mathrm{aer}}$',r'$R^\prime_{\mathrm{low}}$',r'$R^\prime_{\mathrm{mid}}$',r'$R^\prime_{\mathrm{high}}$']
lcolors = ['k',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.75),cm.Purples(.75),cm.Reds(.75)]
lab = ['a','b','c','d','e','f']

plt.figure(figsize=(8,12))
plt.clf()

fs = 18

n = 1
for var in ['Ra','Ra_sfc','Ra_aer','Ra_low','Ra_mid','Ra_high']:
    
    ax = plt.subplot(6,1,n)
    
    plt.plot(np.sin(fbct.lat.values*np.pi/180),dZt[var],c=lcolors[n-1],lw=3,zorder=11)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-1*dZe[var],1*dZe[var],facecolor='.75',label=r'$1\sigma$',zorder=-1)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-2*dZe[var],2*dZe[var],facecolor='.85',label=r'$2\sigma$',zorder=-2)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-3*dZe[var],3*dZe[var],facecolor='.95',label=r'$3\sigma$',zorder=-3)

    plt.plot(np.sin(fbct.lat.values*np.pi/180),3*dZe[var],'k',lw=1,ls='dotted')
    plt.plot(np.sin(fbct.lat.values*np.pi/180),-3*dZe[var],'k',lw=1,ls='dotted')

    plt.plot([-2,2],[0,0],'k--',lw=1,zorder=10)

    plt.xlim(-1,1)
    plt.ylim(-30,30)
    
    plt.xticks(np.sin(np.pi/180*np.arange(-90,91,15)),[r'90$\degree$S',r'',r'',r'',r'30$\degree$S',r'',r'0$\degree$',r'',r'30$\degree$N',r'',r'',r'',r'90$\degree$N'],fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.ylabel('d%s/d%s' % (ylab[n-1],r'$t$'),fontsize=fs)
    
    if n == 6: 
        plt.legend(fontsize=fs-8,ncol=3,loc=3)
        
    ax.text(-.25,1,s='(%s)' % lab[n-1],transform = ax.transAxes,fontsize=fs-2,fontweight='bold')
    
    n += 1

plt.suptitle(r'Zonal trends (W/$\mathrm{m^2}$/decade)',fontsize=fs+2)
    
plt.tight_layout()

plt.savefig(dir_fig+'FigS5.png',dpi=450)
