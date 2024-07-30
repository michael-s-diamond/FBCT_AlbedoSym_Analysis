"""
Code to reproduce Figure S4 in Diamond et al. (2024), ESSOAr

Trend analysis over full CERES EBAF record

Modification history
--------------------
30 March 2023: Michael Diamond, Tallahassee, FL
    -Created
2 July 2024: Michael Diamond, Tallahassee, FL
    -All trends together
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
import os

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'
dir_fig = '/Users/michaeldiamond/Documents/Projects/Albedo_Sym/FBCT_analysis/'

#Load data
ebaf = xr.open_mfdataset(glob(dir_data+'CERES/EBAFed42/EBAF_decomposition_*.nc'))
fTr = xr.open_dataset(dir_data+'CERES/EBAFed42/EBAF_gavg_trends.nc')

"""
Trends
"""

#
###Panel a: Hemispheric trends
#

#Dictionaries to store values
dM = {} #Means
dE = {} #Errors
dN = {} #Degrees of freedom
dC = {} #Confidence (95%)

for reg in fTr.reg.values:
    M = [fTr['Ra_trend'].sel(reg=reg).values,
         fTr['Ra_clr_trend'].sel(reg=reg).values,
         fTr['Ra_sfc_trend'].sel(reg=reg).values,
         fTr['Ra_aer_trend'].sel(reg=reg).values,
         fTr['Ra_cld_trend'].sel(reg=reg).values]
    E = [fTr['Ra_trend_err'].sel(reg=reg).values,
         fTr['Ra_clr_trend_err'].sel(reg=reg).values,
         fTr['Ra_sfc_trend_err'].sel(reg=reg).values,
         fTr['Ra_aer_trend_err'].sel(reg=reg).values,
         fTr['Ra_cld_trend_err'].sel(reg=reg).values]
    N = [fTr['Ra_nu'].sel(reg=reg).values,
         fTr['Ra_clr_nu'].sel(reg=reg).values,
         fTr['Ra_sfc_nu'].sel(reg=reg).values,
         fTr['Ra_aer_nu'].sel(reg=reg).values,
         fTr['Ra_cld_nu'].sel(reg=reg).values]
    dM[reg] = np.array(M)
    dE[reg] = np.array(E)
    dN[reg] = np.array(N)
    dC[reg] = E*stats.t.ppf(.975,N)

markers = ['D','*','o','o','^','s','v']
labels = [r'$R$',r'$R_{\mathrm{clr}}$',r'$R_{\mathrm{sfc}}$',r'$R_{\mathrm{aer}}$',r'$R_{\mathrm{high}}$',r'$R_{\mathrm{mid}}$',r'$R_{\mathrm{low}}$']
colors2 = ['.5','.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.5)]

#
###Panels b-c: Tropical and extratropical trends
#

#Clouds
dTr = {} #Tropical
dEx = {} #Extratropical
    
dTr['cld_trend'] = (fTr['Ra_cld_trend'].sel(reg='NHtr')-fTr['Ra_cld_trend'].sel(reg='SHtr')).values/2
dTr['cld_trend_err'] = np.sqrt((fTr['Ra_cld_trend_err'].sel(reg='NHtr')/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='NHtr')))**2+(fTr['Ra_cld_trend_err'].sel(reg='SHtr')/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='SHtr')))**2).values

dEx['cld_trend'] = (fTr['Ra_cld_trend'].sel(reg='NHex')-fTr['Ra_cld_trend'].sel(reg='SHex')).values/2
dEx['cld_trend_err'] = np.sqrt((fTr['Ra_cld_trend_err'].sel(reg='NHex')/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='NHex')))**2+(fTr['Ra_cld_trend_err'].sel(reg='SHex')/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='SHex')))**2).values
    
#Clear-sky 
dClr = {}

dClr['all_trend'] = fTr['Ra_clr_trend'].sel(reg='NH-SH').values
dClr['all_trend_err'] = (fTr['Ra_clr_trend_err'].sel(reg='NH-SH')*stats.t.ppf(.975,fTr['Ra_clr_nu'].sel(reg='NH-SH'))).values

dClr['aer_trend'] = fTr['Ra_aer_trend'].sel(reg='NH-SH').values 
dClr['aer_trend_err'] = (fTr['Ra_aer_trend_err'].sel(reg='NH-SH')*stats.t.ppf(.975,fTr['Ra_aer_nu'].sel(reg='NH-SH'))).values 

dClr['sfc_trend'] = fTr['Ra_sfc_trend'].sel(reg='NH-SH').values
dClr['sfc_trend_err'] = (fTr['Ra_sfc_trend_err'].sel(reg='NH-SH')*stats.t.ppf(.975,fTr['Ra_sfc_nu'].sel(reg='NH-SH'))).values


"""
Figure
"""

plt.figure(figsize=(9,20))
plt.clf()
fs = 15

#Global trends

markers = ['D','*','o','o','^','s','v']
labels = [r'd($R$)/d$t$',r'd($R_{\mathrm{clr}}$)/d$t$',r'd($R_{\mathrm{sfc}}$)/d$t$',r'd($R_{\mathrm{aer}}$)/d$t$',r'd($R_{\mathrm{cld}}$)/d$t$']
colors = ['.5','.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.5)]

ax1 = plt.subplot(2,1,1)

for i in range(len(dM['NH'])):
    plt.scatter(dM['NH'][i],dM['SH'][i],color=colors[i],s=250,edgecolors='k',lw=1.5,marker=markers[i],label=labels[i],zorder=10)
    
    #NH trend
    if np.sign(dM['NH'][i]-dC['NH'][i]) == np.sign(dM['NH'][i]+dC['NH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]-dC['NH'][i],dM['NH'][i]+dC['NH'][i]],[dM['SH'][i],dM['SH'][i]],c=colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)
    
    #SH trend
    if np.sign(dM['SH'][i]-dC['SH'][i]) == np.sign(dM['SH'][i]+dC['SH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed' 
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i],dM['NH'][i]],[dM['SH'][i]-dC['SH'][i],dM['SH'][i]+dC['SH'][i]],c=colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)

    #Global trend
    if np.sign(dM['global'][i]-dC['global'][i]) == np.sign(dM['global'][i]+dC['global'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]-dC['global'][i],dM['NH'][i]+dC['global'][i]],[dM['SH'][i]-dC['global'][i],dM['SH'][i]+dC['global'][i]],c=colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)
    
    #Symmetry trend
    if np.sign(dM['NH-SH'][i]-dC['NH-SH'][i]) == np.sign(dM['NH-SH'][i]+dC['NH-SH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]+dC['NH-SH'][i]/2,dM['NH'][i]-dC['NH-SH'][i]/2],[dM['SH'][i]-dC['NH-SH'][i]/2,dM['SH'][i]+dC['NH-SH'][i]/2],c=colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)

for e in [.2,.4,.6,.8,1]:
    #Global values
    plt.plot([-10+e,10+e],[10+e,-10+e],'.75',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-10-e,10-e],[10-e,-10-e],'.75',lw=.5,zorder=0,linestyle='dashed')
    #Hemispheric difference
    plt.plot([-10+e,10+e],[-10-e,10-e],'.75',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-10-e,10-e],[-10+e,10+e],'.75',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-10+(e-.1),10+(e-.1)],[-10-(e-.1),10-(e-.1)],'.75',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-10-(e-.1),10-(e-.1)],[-10+(e-.1),10+(e-.1)],'.75',lw=.5,zorder=0,linestyle='dashed')
    #NH and SH
    plt.plot([-10,10],[e,e],'k',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-10,10],[-e,-e],'k',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([e,e],[-10,10],'k',lw=.5,zorder=0,linestyle='dashed')
    plt.plot([-e,-e],[-10,10],'k',lw=.5,zorder=0,linestyle='dashed')

plt.plot([-10,10],[10,-10],'.75',lw=2,zorder=0,linestyle='solid')
plt.plot([-10,10],[-10,10],'.75',lw=2,zorder=0,linestyle='solid')
plt.plot([-10,10],[0,0],'k',lw=2,zorder=0)
plt.plot([0,0],[-10,10],'k',lw=2,zorder=0)
    
plt.legend(frameon=True,fontsize=fs,ncol=1,framealpha=1,loc=2)

plt.xlabel(r'NH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.xticks(fontsize=fs-2)
plt.ylabel(r'SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)

plt.axis([-1.125,.2,-.925,.4])

ax1.text(-.16,1,s='(a)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')


#
### (b) Tropical trends
#

ax2 = plt.subplot(4,1,3)

plt.plot([-1,11],[0,0],'k--',lw=1)

#CLR

plt.plot([-.5,1.5],2*[dClr['all_trend']],lw=3,c='.05',solid_capstyle='butt')
plt.fill_between([-.5,1.5],2*[dClr['all_trend']-dClr['all_trend_err']],2*[dClr['all_trend']+dClr['all_trend_err']],facecolor='.95')

plt.scatter(0,dClr['sfc_trend'],s=250,color=cm.YlOrBr(.33),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(0,dClr['sfc_trend'],yerr=dClr['sfc_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.scatter(1,dClr['aer_trend'],s=250,color=cm.YlOrBr(.67),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(1,dClr['aer_trend'],yerr=dClr['aer_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(.5,.75,'Global\n%s' % r'd($\Delta R_\mathrm{clr}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([1.5,1.5],[-1,1],'k',lw=1,linestyle='solid')

#Tropical clouds

plt.plot([1.5,3.5],2*[dTr['cld_trend']],lw=3,color=cm.Reds(.95),solid_capstyle='butt')
plt.fill_between([1.5,3.5],2*[dTr['cld_trend']-dTr['cld_trend_err']],2*[dTr['cld_trend']+dTr['cld_trend_err']],facecolor=cm.Reds(.05))

plt.text(2.5,.75,'Tropical\n%s' % r'0.5 d($\Delta R_\mathrm{cld}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([3.5,3.5],[-1,1],'k',lw=1,linestyle='dotted')
    
#Extratropical clouds
    
plt.plot([3.5,5.5],2*[dEx['cld_trend']],lw=3,color=cm.Blues(.95),solid_capstyle='butt')
plt.fill_between([3.5,5.5],2*[dEx['cld_trend']-dTr['cld_trend_err']],2*[dEx['cld_trend']+dEx['cld_trend_err']],facecolor=cm.Blues(.05))
    
plt.text(4.5,.75,'Extratropical\n%s' % r'0.5 d($\Delta R_\mathrm{cld}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.xlim(-.5,5.5)
plt.ylim(-.8,.8)

plt.ylabel(r'NH-SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks([0,1,2.5,4.5],['sfc','aer','Tropical cld','Extratropical cld'],fontsize=fs)

ax2.text(-.16,1.025,s='(b)',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')



plt.savefig(dir_fig+'FigS4.png',dpi=450)

