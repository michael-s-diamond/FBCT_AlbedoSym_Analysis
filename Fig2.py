"""
Code to reproduce Figure 2 in Diamond et al. (2024), ESSOAr

Trend analysis over full CERES FBCT record

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
fbct = xr.open_mfdataset(glob(dir_data+'CERES/FluxByCldType/FBCT_decomp*nc'))
fbct['Ra_low'] = fbct['Ra_Cu']+fbct['Ra_Sc']+fbct['Ra_St']
fbct['Ra_mid'] = fbct['Ra_Ac']+fbct['Ra_As']+fbct['Ra_Ns']
fbct['Ra_high'] = fbct['Ra_Ci']+fbct['Ra_Cs']+fbct['Ra_Cb']

fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']
fTr['Ra_low'] = fTr['Ra_cld'][:3].sum(axis=0)
fTr['Ra_mid'] = fTr['Ra_cld'][3:6].sum(axis=0)
fTr['Ra_high'] = fTr['Ra_cld'][6:].sum(axis=0)

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

low = np.arange(len(fTr.cld))<=2
mid = np.logical_and(np.arange(len(fTr.cld))>2,np.arange(len(fTr.cld))<=5)
high = np.arange(len(fTr.cld))>5

for reg in fTr.reg.values:
    M = [fTr['Ra_trend'].sel(reg=reg).values,
         fTr['Ra_clr_trend'].sel(reg=reg).values,
         fTr['Ra_sfc_trend'].sel(reg=reg).values,
         fTr['Ra_aer_trend'].sel(reg=reg).values]+[fTr['Ra_cld_trend'][high].sel(reg=reg).sum().values]+[fTr['Ra_cld_trend'][mid].sel(reg=reg).sum().values]+[fTr['Ra_cld_trend'][low].sel(reg=reg).sum().values]
    E = [fTr['Ra_trend_err'].sel(reg=reg).values,
         fTr['Ra_clr_trend_err'].sel(reg=reg).values,
         fTr['Ra_sfc_trend_err'].sel(reg=reg).values,
         fTr['Ra_aer_trend_err'].sel(reg=reg).values]+[np.sqrt(np.sum(fTr['Ra_cld_trend_err'][high].sel(reg=reg).values**2))]+[np.sqrt(np.sum(fTr['Ra_cld_trend_err'][mid].sel(reg=reg).values**2))]+[np.sqrt(np.sum(fTr['Ra_cld_trend_err'][low].sel(reg=reg).values**2))]
    N = [fTr['Ra_nu'].sel(reg=reg).values,
         fTr['Ra_clr_nu'].sel(reg=reg).values,
         fTr['Ra_sfc_nu'].sel(reg=reg).values,
         fTr['Ra_aer_nu'].sel(reg=reg).values]+[fTr['Ra_cld_nu'][high].sel(reg=reg).min().values]+[fTr['Ra_cld_nu'][mid].sel(reg=reg).min().values]+[fTr['Ra_cld_nu'][low].sel(reg=reg).min().values]
    dM[reg] = np.array(M)
    dE[reg] = np.array(E)
    dN[reg] = np.array(N)
    dC[reg] = E*stats.t.ppf(.975,N)

    
#
###Panels b-c: Tropical and extratropical trends
#

dTr = {} #Tropical
dEx = {} #Extratropical

for cld in cldtypes:
    
    dTr[cld] = fTr['Ra_cld'].sel(reg='NHtr',cld=cld)-fTr['Ra_cld'].sel(reg='SHtr',cld=cld)
    dEx[cld] = fTr['Ra_cld'].sel(reg='NHex',cld=cld)-fTr['Ra_cld'].sel(reg='SHex',cld=cld)
    
    #Trends for tropics and extratropics (1/2 weighting for global effect)
    dTr[cld+'_trend'] = (fTr['Ra_cld_trend'].sel(reg='NHtr',cld=cld)-fTr['Ra_cld_trend'].sel(reg='SHtr',cld=cld)).values/2
    dEx[cld+'_trend'] = (fTr['Ra_cld_trend'].sel(reg='NHex',cld=cld)-fTr['Ra_cld_trend'].sel(reg='SHex',cld=cld)).values/2
    
    dTr[cld+'_trend_err'] = np.sqrt((fTr['Ra_cld_trend_err'].sel(reg='NHtr',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='NHtr',cld=cld)))**2+(fTr['Ra_cld_trend_err'].sel(reg='SHtr',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='SHtr',cld=cld)))**2).values
    dEx[cld+'_trend_err'] = np.sqrt((fTr['Ra_cld_trend_err'].sel(reg='NHex',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='NHex',cld=cld)))**2+(fTr['Ra_cld_trend_err'].sel(reg='SHex',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_cld_nu'].sel(reg='SHex',cld=cld)))**2).values

#Tropics

dTr['high'] = dTr['Ci']+dTr['Cs']+dTr['Cb']
dTr['high_trend'] = dTr['Ci_trend']+dTr['Cs_trend']+dTr['Cb_trend']
dTr['high_trend_err'] = np.sqrt(dTr['Ci_trend_err']**2+dTr['Cs_trend_err']**2+dTr['Cb_trend_err']**2)

dTr['mid'] = dTr['Ac']+dTr['As']+dTr['Ns']
dTr['mid_trend'] = dTr['Ac_trend']+dTr['As_trend']+dTr['Ns_trend']
dTr['mid_trend_err'] = np.sqrt(dTr['Ac_trend_err']**2+dTr['As_trend_err']**2+dTr['Ns_trend_err']**2)

dTr['low'] = dTr['Cu']+dTr['Sc']+dTr['St']
dTr['low_trend'] = dTr['Cu_trend']+dTr['Sc_trend']+dTr['St_trend']
dTr['low_trend_err'] = np.sqrt(dTr['Cu_trend_err']**2+dTr['Sc_trend_err']**2+dTr['St_trend_err']**2)

#Extratropics

dEx['high'] = dEx['Ci']+dEx['Cs']+dEx['Cb']
dEx['high_trend'] = dEx['Ci_trend']+dEx['Cs_trend']+dEx['Cb_trend']
dEx['high_trend_err'] = np.sqrt(dEx['Ci_trend_err']**2+dEx['Cs_trend_err']**2+dEx['Cb_trend_err']**2)

dEx['mid'] = dEx['Ac']+dEx['As']+dEx['Ns']
dEx['mid_trend'] = dEx['Ac_trend']+dEx['As_trend']+dEx['Ns_trend']
dEx['mid_trend_err'] = np.sqrt(dEx['Ac_trend_err']**2+dEx['As_trend_err']**2+dEx['Ns_trend_err']**2)

dEx['low'] = dEx['Cu']+dEx['Sc']+dEx['St']
dEx['low_trend'] = dEx['Cu_trend']+dEx['Sc_trend']+dEx['St_trend']
dEx['low_trend_err'] = np.sqrt(dEx['Cu_trend_err']**2+dEx['Sc_trend_err']**2+dEx['St_trend_err']**2)

#Clear-sky 

dClr = {}

dClr['all_trend'] = fTr['Ra_clr_trend'].sel(reg='NH-SH').values #Both surface and atmosphere
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
labels = [r'd($R^\prime$)/d$t$',r'd($R^\prime_{\mathrm{clr}}$)/d$t$',r'd($R^\prime_{\mathrm{sfc}}$)/d$t$',r'd($R^\prime_{\mathrm{aer}}$)/d$t$',r'd($R^\prime_{\mathrm{high}}$)/d$t$',r'd($R^\prime_{\mathrm{mid}}$)/d$t$',r'd($R^\prime_{\mathrm{low}}$)/d$t$']
face_colors = ['.5','.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Purples(.25),cm.Blues(.25)]
edge_colors = 4*['k']+[cm.Reds(.75),cm.Purples(.75),cm.Blues(.75)]
err_colors = ['.5','.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.5),cm.Purples(.5),cm.Blues(.5)]


ax1 = plt.subplot(2,1,1)

for i in range(len(dM['NH'])):
    plt.scatter(dM['NH'][i],dM['SH'][i],color=face_colors[i],s=250,edgecolors=edge_colors[i],lw=1.5,marker=markers[i],label=labels[i],zorder=10)
    
    #NH trend
    if np.sign(dM['NH'][i]-dC['NH'][i]) == np.sign(dM['NH'][i]+dC['NH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]-dC['NH'][i],dM['NH'][i]+dC['NH'][i]],[dM['SH'][i],dM['SH'][i]],c=err_colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)
    
    #SH trend
    if np.sign(dM['SH'][i]-dC['SH'][i]) == np.sign(dM['SH'][i]+dC['SH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed' 
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i],dM['NH'][i]],[dM['SH'][i]-dC['SH'][i],dM['SH'][i]+dC['SH'][i]],c=err_colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)

    #Global trend
    if np.sign(dM['global'][i]-dC['global'][i]) == np.sign(dM['global'][i]+dC['global'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]-dC['global'][i],dM['NH'][i]+dC['global'][i]],[dM['SH'][i]-dC['global'][i],dM['SH'][i]+dC['global'][i]],c=err_colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)
    
    #Symmetry trend
    if np.sign(dM['NH-SH'][i]-dC['NH-SH'][i]) == np.sign(dM['NH-SH'][i]+dC['NH-SH'][i]): 
        ls = 'solid'
        marker = 'o'
    else: 
        ls = 'dashed'
        marker = r'$\mathrm{o}$'
    plt.plot([dM['NH'][i]+dC['NH-SH'][i]/2,dM['NH'][i]-dC['NH-SH'][i]/2],[dM['SH'][i]-dC['NH-SH'][i]/2,dM['SH'][i]+dC['NH-SH'][i]/2],c=err_colors[i],lw=2,zorder=9,marker=marker,linestyle=ls)

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

plt.axis([-1.025,.3,-.825,.5])

ax1.text(-.16,1,s='(a)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')


#
### (b) Tropical trends
#

dColors = {'Cu' : cm.Blues(.25), 'Sc' : cm.Blues(.5), 'St' : cm.Blues(.75), 'Ac' : cm.Purples(.25), 'As' : cm.Purples(.5), 'Ns' : cm.Purples(.75), 'Ci' : cm.Reds(.25), 'Cs' : cm.Reds(.5), 'Cb' : cm.Reds(.75)}
dMarkers = {'Cu' : 'v', 'Sc' : 'v', 'St' : 'v', 'Ac' : 's', 'As' : 's', 'Ns' : 's', 'Ci' : '^', 'Cs' : '^', 'Cb' : '^'}

ax2 = plt.subplot(4,1,3)

plt.plot([-1,11],[0,0],'k--',lw=1)

#CLR

plt.plot([-.5,1.5],2*[dClr['all_trend']],lw=3,c='.05',solid_capstyle='butt')
plt.fill_between([-.5,1.5],2*[dClr['all_trend']-dClr['all_trend_err']],2*[dClr['all_trend']+dClr['all_trend_err']],facecolor='.95')

plt.scatter(0,dClr['sfc_trend'],s=250,color=cm.YlOrBr(.33),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(0,dClr['sfc_trend'],yerr=dClr['sfc_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.scatter(1,dClr['aer_trend'],s=250,color=cm.YlOrBr(.67),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(1,dClr['aer_trend'],yerr=dClr['aer_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(.5,.6,'Global\n%s' % r'd($\Delta R_\mathrm{clr}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([1.5,1.5],[-1,1],'k',lw=1,linestyle='solid')

#High clouds

plt.plot([1.5,4.5],2*[dTr['high_trend']],lw=3,color=cm.Reds(.95),solid_capstyle='butt')
plt.fill_between([1.5,4.5],2*[dTr['high_trend']-dTr['high_trend_err']],2*[dTr['high_trend']+dTr['high_trend_err']],facecolor=cm.Reds(.05))

n = 1
for cld in ['Ci','Cs','Cb']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(3,.6,'Tropical\n%s' % r'0.5 d($\Delta R_\mathrm{high}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([4.5,4.5],[-1,1],'k',lw=1,linestyle='dotted')
    
#Mid-level clouds
    
plt.plot([4.5,7.5],2*[dTr['mid_trend']],lw=3,color=cm.Purples(.95),solid_capstyle='butt')
plt.fill_between([4.5,7.5],2*[dTr['mid_trend']-dTr['mid_trend_err']],2*[dTr['mid_trend']+dTr['mid_trend_err']],facecolor=cm.Purples(.05))

for cld in ['Ac','As','Ns']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(6,.6,'Tropical\n%s' % r'0.5 d($\Delta R_\mathrm{mid}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([7.5,7.5],[-1,1],'k',lw=1,linestyle='dotted')

#Low clouds

plt.plot([7.5,10.5],2*[dTr['low_trend']],lw=3,color=cm.Blues(.95),solid_capstyle='butt')
plt.fill_between([7.5,10.5],2*[dTr['low_trend']-dTr['low_trend_err']],2*[dTr['low_trend']+dTr['low_trend_err']],facecolor=cm.Blues(.05))

for cld in ['Cu','Sc','St']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(9,.6,'Tropical\n%s' % r'0.5 d($\Delta R_\mathrm{low}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.xlim(-.5,10.5)
plt.ylim(-.7,.7)

plt.ylabel(r'NH-SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(11),['sfc','aer','Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St'],fontsize=fs)

ax2.text(-.16,1.025,s='(b)',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')


#
### (c) Extratropical trends
#

ax3 = plt.subplot(4,1,4)

plt.plot([-1,11],[0,0],'k--',lw=1)

#CLR

plt.plot([-.5,1.5],2*[dClr['all_trend']],lw=3,c='.05',solid_capstyle='butt')
plt.fill_between([-.5,1.5],2*[dClr['all_trend']-dClr['all_trend_err']],2*[dClr['all_trend']+dClr['all_trend_err']],facecolor='.95')

plt.scatter(0,dClr['sfc_trend'],s=250,color=cm.YlOrBr(.33),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(0,dClr['sfc_trend'],yerr=dClr['sfc_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.scatter(1,dClr['aer_trend'],s=250,color=cm.YlOrBr(.67),marker='o',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(1,dClr['aer_trend'],yerr=dClr['aer_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(.5,.6,'Global\n%s' % r'd($\Delta R_\mathrm{clr}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([1.5,1.5],[-1,1],'k',lw=1,linestyle='solid')

#High clouds

plt.plot([1.5,4.5],2*[dEx['high_trend']],lw=3,color=cm.Reds(.95),solid_capstyle='butt')
plt.fill_between([1.5,4.5],2*[dEx['high_trend']-dEx['high_trend_err']],2*[dEx['high_trend']+dEx['high_trend_err']],facecolor=cm.Reds(.05))

n = 1
for cld in ['Ci','Cs','Cb']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(3,.6,'Extratropical\n%s' % r'0.5 d($\Delta R_\mathrm{high}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([4.5,4.5],[-1,1],'k',lw=1,linestyle='dotted')

#Mid-level clouds

plt.plot([4.5,7.5],2*[dEx['mid_trend']],lw=3,color=cm.Purples(.95),solid_capstyle='butt')
plt.fill_between([4.5,7.5],2*[dEx['mid_trend']-dEx['mid_trend_err']],2*[dEx['mid_trend']+dEx['mid_trend_err']],facecolor=cm.Purples(.05))

for cld in ['Ac','As','Ns']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(6,.6,'Extratropical\n%s' % r'0.5 d($\Delta R_\mathrm{mid}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([7.5,7.5],[-1,1],'k',lw=1,linestyle='dotted')

#Low clouds

plt.plot([7.5,10.5],2*[dEx['low_trend']],lw=3,color=cm.Blues(.95),solid_capstyle='butt')
plt.fill_between([7.5,10.5],2*[dEx['low_trend']-dEx['low_trend_err']],2*[dEx['low_trend']+dEx['low_trend_err']],facecolor=cm.Blues(.05))

for cld in ['Cu','Sc','St']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(9,.6,'Extratropical\n%s' % r'0.5 d($\Delta R_\mathrm{low}^\prime$)/d$t$',ha='center',va='top',fontsize=fs-2)

plt.xlim(-.5,10.5)
plt.ylim(-.7,.7)

plt.ylabel(r'NH-SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(11),['sfc','aer','Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St'],fontsize=fs)

ax3.text(-.16,1.025,s='(c)',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')


plt.savefig(dir_fig+'Fig2.png',dpi=450)

