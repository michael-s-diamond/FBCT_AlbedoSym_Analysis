"""
Code to reproduce Figure 1 in Diamond et al. (2023), GRL

Climatology and trends of Earth's hemispheric albedo symmetry by cloud type

Modification history
--------------------
16 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
"""

#Import libraries
import numpy as np
import xarray as xr
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
fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']

"""
Create plot for Figure 1 (global values)
"""

#
###Panel a: Waterfall plot of climatology
#

Mwts = np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])
climo = []
climo.append(np.average(fTr['Rc'].sel(reg='NH'),weights=Mwts)) #NH average
climo.append(np.average(fTr['Rc_sfc'].sel(reg='NH-SH'),weights=Mwts)) #NH-SH sfc average
for ct in ['clr']+cldtypes:#NH-SH atm averages
    climo.append(np.average(fTr['Rc_atm'].sel(cld=ct,reg='NH-SH'),weights=Mwts))
climo.append(np.average(fTr['Rc'].sel(reg='SH'),weights=Mwts)) #SH average

ybase = [0]+[np.sum(climo[:i]) for i in range(1,len(climo)-1)]+[0]

colors0 = ['.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75)]


#
###Panel b: Zonal differences
#

dZ = {}
for var in ['Rc','Rc_sfc','Rc_atm_clr']+['Rc_atm_%s' % ct for ct in cldtypes]:
    dZ[var] = np.average(fbct[var],weights=Mwts[:,np.newaxis,np.newaxis]*np.ones(fbct[var].shape),axis=(0,-1))

colors1 = [cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75)]


#
###Panel c: Hemispheric trends
#

#Dictionaries to store values
dM = {} #Means
dE = {} #Errors
dN = {} #Degrees of freedom
dC = {} #Confidence (95%)

for reg in fTr.reg.values:
    M = [fTr['Ra_trend'].sel(reg=reg).values,fTr['Ra_sfc_trend'].sel(reg=reg).values,fTr['Ra_atm_trend'].sel(cld='clr',reg=reg).values]+[fTr['Ra_atm_trend'].sel(cld=cld,reg=reg).values for cld in cldtypes]+[fTr['S*Aa_clr_trend'].sel(reg=reg).values,fTr['S*Aa_sfc_trend'].sel(reg=reg).values,fTr['S*Aa_atm_trend'].sel(cld='clr',reg=reg).values]
    E = [fTr['Ra_trend_err'].sel(reg=reg).values,fTr['Ra_sfc_trend_err'].sel(reg=reg).values,fTr['Ra_atm_trend_err'].sel(cld='clr',reg=reg).values]+[fTr['Ra_atm_trend_err'].sel(cld=cld,reg=reg).values for cld in cldtypes]+[fTr['S*Aa_clr_trend_err'].sel(reg=reg).values,fTr['S*Aa_sfc_trend_err'].sel(reg=reg).values,fTr['S*Aa_atm_trend_err'].sel(cld='clr',reg=reg).values]
    N = [fTr['Ra_nu'].sel(reg=reg).values,fTr['Ra_sfc_nu'].sel(reg=reg).values,fTr['Ra_atm_nu'].sel(cld='clr',reg=reg).values]+[fTr['Ra_atm_nu'].sel(cld=cld,reg=reg).values for cld in cldtypes]+[fTr['S*Aa_clr_nu'].sel(reg=reg).values,fTr['S*Aa_sfc_nu'].sel(reg=reg).values,fTr['S*Aa_atm_nu'].sel(cld='clr',reg=reg).values]
    dM[reg] = np.array(M)
    dE[reg] = np.array(E)
    dN[reg] = np.array(N)
    dC[reg] = E*stats.t.ppf(.975,N)

Gerr = dC['global'][0]
Derr = dC['NH-SH'][0]
Gclrerr = dC['global'][-3]
Dclrerr = dC['NH-SH'][-3]

markers = ['D','o','o','^','^','^','s','s','s','v','v','v','*','*','*']
labels = [r'$R$',r'$R_{\mathrm{sfc}}$',r'$R_{\mathrm{aer}}$']+[r'$R_{\mathrm{%s}}$' % ct for ct in cldtypes]+[r'${S}{A}_\mathrm{clr}$',r'${S}{A}_{\mathrm{sfc}}$',r'${S}{A}_{\mathrm{aer}}$']
colors2 = ['.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75),'.5',cm.YlOrBr(.33),cm.YlOrBr(.67)]


#
###Plot values
#

plt.figure(figsize=(20,9.5))
plt.clf()
fs = 16

#Waterfall plot

ax1 = plt.subplot(2,2,1)

bars = plt.bar(np.arange(13),climo,bottom=ybase,width=.67,color=colors0,capsize=4,edgecolor='k',lw=2)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

xline = [climo[0],climo[0]]
for i in range(2,12):
    xline.append(ybase[i])
    xline.append(ybase[i])
xline.append(climo[-1])
xline.append(climo[-1])

plt.plot([0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12],xline,'k',lw=2,zorder=0)

plt.text(0,climo[0]+.75,'%.1f' % climo[0],fontsize=fs-2,ha='center',weight='black')
for i in range(1,12):
    if climo[i] > 0: 
        label = '+%.1f' % climo[i]
        offset = .75
    elif climo[i] < 0:
        label = '%.1f' % climo[i]
        offset = -.75
    plt.text(i,climo[i]+ybase[i]+offset,label,fontsize=fs-2,ha='center',va='center')
plt.text(12,climo[-1]+.75,'%.1f' % climo[-1],fontsize=fs-2,ha='center',weight='black')
    
plt.ylim(94,108)

plt.ylabel(r'$\overline{R}$ ($\mathrm{W/m^{2}}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(13),['NH','sfc','aer','Ci', 'Cs', 'Cb','Ac','As','Ns','Cu','Sc','St','SH'],fontsize=fs-2)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.text(-.15,1,s='a',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#Zonal differences

ax2 = plt.subplot(2,2,3)

ax2.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),np.array(dZ['Rc'][90:]-dZ['Rc'][:90][::-1]),'k',lw=5,label='Total')
base2 = np.array(dZ['Rc_sfc'][90:]-dZ['Rc_sfc'][:90][::-1]) #Surface
ax2.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),base2,c=colors1[0],zorder=1-i,lw=5,label='sfc',linestyle='dotted')

for i in range(2,len(dZ.keys())):
    
    if i == 2:
        hatch = '/'
        ls = 'dashed'
    else:
        hatch = ''
        ls = 'solid'
    
    ct = list(dZ.keys())[i]
    
    base2 = np.array(dZ[ct][90:]-dZ[ct][:90][::-1])
        
    if i == 2: label = 'aer'
    else: label = ct.split('_')[-1]
    ax2.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),base2,c=colors1[i-1],zorder=1-i,lw=3,label=label,linestyle=ls)

ax2.plot(np.sin(fbct['lat'].values*np.pi/180),0*np.sin(fbct['lat'].values*np.pi/180),'k',linestyle='dashed',lw=2,zorder=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axis([0,1,-50,50])
ax2.tick_params(labelsize=fs-2)
xticks = np.arange(0,91,15)
xticklabs = [r'0$\degree$',r'15$\degree$',r'30$\degree$',r'45$\degree$',r'60$\degree$','',r'90$\degree$']
ax2.set_xticks(np.sin(xticks*np.pi/180))
ax2.set_xticklabels(xticklabs)
ax2.set_ylabel(r'$\Delta \overline{R}$ ($\mathrm{W/m^2}$)',fontsize=fs)
ax2.legend(frameon=False,fontsize=fs-2,ncol=4,loc=3)
ax2.text(-.15,1,s='b',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

#Trends

ax3 = plt.subplot(1,2,2)

for i in range(len(dM['NH'])):
    plt.scatter(dM['NH'][i],dM['SH'][i],color=colors2[i],s=200,edgecolors='k',lw=1,marker=markers[i],label=labels[i],zorder=10)
    plt.errorbar(dM['NH'][i],dM['SH'][i],xerr=dC['NH'][i],yerr=dC['SH'][i],color=colors2[i],capsize=4,lw=2,zorder=9,capthick=2)

plt.plot([dM['NH'][0]-Gerr,dM['NH'][0]+Gerr],[dM['SH'][0]-Gerr,dM['SH'][0]+Gerr],c=colors2[0],lw=2,zorder=9,marker='o',linestyle='dashed')
plt.plot([dM['NH'][0]+Derr/2,dM['NH'][0]-Derr/2],[dM['SH'][0]-Derr/2,dM['SH'][0]+Derr/2],c=colors2[0],lw=2,zorder=9,marker='o',linestyle='dashed')

plt.plot([dM['NH'][-3]-Gclrerr,dM['NH'][-3]+Gclrerr],[dM['SH'][-3]-Gclrerr,dM['SH'][-3]+Gclrerr],c=colors2[0],lw=2,zorder=9,marker='o',linestyle='dashed')
plt.plot([dM['NH'][-3]+Dclrerr/2,dM['NH'][-3]-Dclrerr/2],[dM['SH'][-3]-Dclrerr/2,dM['SH'][-3]+Dclrerr/2],c=colors2[0],lw=2,zorder=9,marker='o',linestyle='dashed')

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

plt.plot([-10,10],[10,-10],'.75',lw=1,zorder=0,linestyle='solid')
plt.plot([-10,10],[-10,10],'.75',lw=1,zorder=0,linestyle='solid')
plt.plot([-10,10],[0,0],'k',lw=1,zorder=0)
plt.plot([0,0],[-10,10],'k',lw=1,zorder=0)
    
plt.legend(frameon=True,fontsize=fs-2,ncol=5,framealpha=1,loc=2)

plt.xlabel(r'NH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.xticks(fontsize=fs-2)
plt.ylabel(r'SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)

plt.axis([-1.025,.3,-.825,.5])

ax3.text(-.15,1,s='c',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')

plt.savefig(dir_fig+'Fig1.png',dpi=150)
plt.savefig(dir_fig+'Fig1.eps')
















