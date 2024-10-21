"""
Code to reproduce EBAF-version of Figure 1ab in Diamond et al. (2024), GRL

Climatology of Earth's hemispheric albedo symmetry

Modification history
--------------------
16 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
07 May 2024: Michael Diamond, Tallahassee, FL
    -Simplified cloud definitions
    -Adding more FBCT summary panels to figure
21 June 2024: Michael Diamond, Tallahassee, FL
    -Simplified clear-sky decomposition
    -Error bars for all components (low/mid/high cloud aggregates)
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
ebaf = xr.open_mfdataset(glob(dir_data+'CERES/EBAFed42/EBAF_decomposition_*.nc'))
fTr = xr.open_dataset(dir_data+'CERES/EBAFed42/EBAF_gavg_trends.nc')

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
climo.append(np.average(fTr['Rc_aer'].sel(reg='NH-SH'),weights=Mwts)) #NH-SH aer average
climo.append(np.average(fTr['Rc_cld'].sel(reg='NH-SH'),weights=Mwts)) #NH-SH cld average
climo.append(np.average(fTr['Rc'].sel(reg='SH'),weights=Mwts)) #SH average

ybase = [0]+[np.sum(climo[:i]) for i in range(1,len(climo)-1)]+[0]


Dtr_sfc = (np.average(fTr['Rc_sfc'].sel(reg='NHtr'),weights=Mwts)-np.average(fTr['Rc_sfc'].sel(reg='SHtr'),weights=Mwts))/2
Dex_sfc = (np.average(fTr['Rc_sfc'].sel(reg='NHex'),weights=Mwts)-np.average(fTr['Rc_sfc'].sel(reg='SHex'),weights=Mwts))/2

Dtr_aer = (np.average(fTr['Rc_aer'].sel(reg='NHtr'),weights=Mwts)-np.average(fTr['Rc_aer'].sel(reg='SHtr'),weights=Mwts))/2
Dex_aer = (np.average(fTr['Rc_aer'].sel(reg='NHex'),weights=Mwts)-np.average(fTr['Rc_aer'].sel(reg='SHex'),weights=Mwts))/2

Dtr_cld = (np.average(fTr['Rc_cld'].sel(reg='NHtr'),weights=Mwts)-np.average(fTr['Rc_cld'].sel(reg='SHtr'),weights=Mwts))/2
Dex_cld = (np.average(fTr['Rc_cld'].sel(reg='NHex'),weights=Mwts)-np.average(fTr['Rc_cld'].sel(reg='SHex'),weights=Mwts))/2


colors0 = ['.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.5)]


#
###Panel b: Zonal differences
#

dZ = {}
for var in ['Rc','Rc_sfc','Rc_aer','Rc_cld']:
    dZ[var] = np.average(ebaf[var],weights=Mwts[:,np.newaxis,np.newaxis]*np.ones(ebaf[var].shape),axis=(0,-1))

colors1 = [cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.5)]


#
###Plot values
#

plt.figure(figsize=(9,9))
plt.clf()
fs = 16

#Waterfall plot

ax1 = plt.subplot(2,1,1)

bars = plt.bar(np.arange(5),climo,bottom=ybase,width=.67,color=colors0,capsize=4,edgecolor='k',lw=2)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

xline = [climo[0],climo[0]]
for i in range(2,4):
    xline.append(ybase[i])
    xline.append(ybase[i])
xline.append(climo[-1])
xline.append(climo[-1])

plt.plot([0,1,1,2,2,3,3,4],xline,'k',lw=2,zorder=0)

plt.text(0,climo[0]+.75,'%.1f' % climo[0],fontsize=fs-2,ha='center',weight='black')
for i in range(1,4):
    if climo[i] > 0: 
        label = '+%.1f' % climo[i]
        offset = .75
    elif climo[i] < 0:
        label = '%.1f' % climo[i]
        offset = -.75
    plt.text(i,climo[i]+ybase[i]+offset,label,fontsize=fs-2,ha='center',va='center')
plt.text(4,climo[-1]+.75,'%.1f' % climo[-1],fontsize=fs-2,ha='center',weight='black')

#Tr versus ex breakdown
plt.scatter(1-.18,ybase[1]+Dtr_sfc,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tr')
plt.scatter(1+.18,ybase[1]+Dex_sfc,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Ex')

plt.scatter(2-.18,ybase[2]+Dtr_aer,s=100,facecolor='k',edgecolors='w',lw=1,marker='P')
plt.scatter(2+.18,ybase[2]+Dex_aer,s=100,facecolor='k',edgecolors='w',lw=1,marker='X')

plt.scatter(3-.18,ybase[3]+Dtr_cld,s=100,facecolor='k',edgecolors='w',lw=1,marker='P')
plt.scatter(3+.18,ybase[3]+Dex_cld,s=100,facecolor='k',edgecolors='w',lw=1,marker='X')

plt.legend(fontsize=fs-2)

plt.ylim(94,106)

plt.ylabel(r'$\overline{R}$ ($\mathrm{W/m^{2}}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(5),['NH','sfc','aer','cld','SH'],fontsize=fs-2)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.text(-.15,1,s='(a)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#Zonal differences

ax2 = plt.subplot(2,1,2)

ax2.plot(np.sin(ebaf['lat'].values[90:]*np.pi/180),np.array(dZ['Rc'][90:]-dZ['Rc'][:90][::-1]),'k',lw=5,label='Total')
base2 = np.array(dZ['Rc_sfc'][90:]-dZ['Rc_sfc'][:90][::-1]) #Surface
ax2.plot(np.sin(ebaf['lat'].values[90:]*np.pi/180),base2,c=colors1[0],zorder=1-i,lw=5,label='sfc',linestyle='dotted')

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
    ax2.plot(np.sin(ebaf['lat'].values[90:]*np.pi/180),base2,c=colors1[i-1],zorder=1-i,lw=3,label=label,linestyle=ls)

ax2.plot(np.sin(ebaf['lat'].values*np.pi/180),0*np.sin(ebaf['lat'].values*np.pi/180),'k',linestyle='dashed',lw=2,zorder=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axis([0,1,-40,20])
ax2.tick_params(labelsize=fs-2)
xticks = np.arange(0,91,15)
xticklabs = [r'0$\degree$',r'15$\degree$',r'30$\degree$',r'45$\degree$',r'60$\degree$','',r'90$\degree$']
ax2.set_xticks(np.sin(xticks*np.pi/180))
ax2.set_xticklabels(xticklabs)
ax2.set_ylabel(r'$\Delta \overline{R}$ ($\mathrm{W/m^2}$)',fontsize=fs)
ax2.legend(frameon=False,fontsize=fs-2,ncol=4,loc=3)
ax2.text(-.15,1,s='(b)',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

plt.savefig(dir_fig+'FigS3.png',dpi=450)
