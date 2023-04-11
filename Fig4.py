"""
Code to reproduce Figure 4 in Diamond et al. (2023), GRL

"Natural experiment" from COVID aerosol decline

Modification history
--------------------
10 April 2023: Michael Diamond, Tallahassee, FL
    -Created
"""

#Import libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
import os

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'
dir_fig = '/Users/michaeldiamond/Documents/Projects/Albedo_Sym/FBCT_analysis/'

#Load data
fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fTr.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

#
###Annual averages of clear-sky atmospheric reflection
#

dAnn = {} #Storing annual time series

for reg in ['global','NH','SH','NH-SH']:
    dAnn[reg] = np.array([np.average(fTr['S*Aa_atm'].sel(cld='clr',reg=reg)[fTr.time.dt.year==yr],weights=Mwts[fTr.time.dt.year==yr]) for yr in range(2003,2023)])


"""
Analyze potential COVID anomalies
"""

cov19 = fTr.time.dt.year==2020

dCov = {reg : {} for reg in fTr.reg.values}

for reg in fTr.reg.values:
    
    #Clear-sky
    dCov[reg]['SAclr'] = np.average(fTr['S*Aa_clr'].sel(reg=reg)[cov19],weights=Mwts[cov19])
    
    dCov[reg]['SAsfc'] = np.average(fTr['S*Aa_sfc'].sel(reg=reg)[cov19],weights=Mwts[cov19])
    
    dCov[reg]['SAaer'] = np.average(fTr['S*Aa_atm'].sel(cld='clr',reg=reg)[cov19],weights=Mwts[cov19])
    
    #All-sky
    dCov[reg]['R'] = np.average(fTr['Ra'].sel(reg=reg)[cov19],weights=Mwts[cov19])
    
    dCov[reg]['Rsfc'] = np.average(fTr['Ra_sfc'].sel(reg=reg)[cov19],weights=Mwts[cov19])
    
    dCov[reg]['Raer'] = np.average(fTr['Ra_atm'].sel(cld='clr',reg=reg)[cov19],weights=Mwts[cov19])
    
    for cld in cldtypes:
        dCov[reg]['R%s' % cld] = np.average(fTr['Ra_atm'].sel(cld=cld,reg=reg)[cov19],weights=Mwts[cov19])

        
        
"""
Create plot
"""     

colors = ['k',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75)]
xticks = ['Total','sfc','aer']+cldtypes

#
###Plot
#
plt.figure(figsize=(12,18))
plt.clf()
fs = 15       

#NH SAaer

ax1 = plt.subplot(5,1,1)

plt.plot(fTr.time,fTr['S*Aa_atm'].sel(cld='clr',reg='NH'),c='k',lw=3)
plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=1)

plt.scatter(fTr.time[12::12].values,dAnn['NH'],s=100,color=cm.Greens(.33),edgecolors='k',lw=1,zorder=10)
plt.scatter(np.datetime64('2020-07-15'),dAnn['NH'][-3],s=150,color=cm.Greens(.67),edgecolors='k',lw=1,zorder=11)

plt.plot(2*[np.datetime64('2020-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2020-12-31')],[-2,2],'k',ls='dotted',lw=1)

plt.fill_between([np.datetime64('2020-01-01'),np.datetime64('2020-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.01),zorder=0)

plt.text(np.datetime64('2020-07-01'),1.75,r'COV',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2020-07-01'),-1.95,'%.2f' % (dAnn['NH'][-3]),ha='center',va='bottom',fontsize=fs-2)

plt.xlim(fTr.time[0].values,fTr.time[-1].values)
plt.ylim(-2,2)

plt.ylabel(r'NH $(S A_\mathrm{aer})^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(fontsize=fs-2)

ax1.text(-.125,1.025,s='a',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#NH-SH SAsfc

ax2 = plt.subplot(5,1,2)

plt.plot(fTr.time,fTr['S*Aa_atm'].sel(cld='clr',reg='NH-SH'),c='k',lw=3)
plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=1)

plt.scatter(fTr.time[12::12].values,dAnn['NH-SH'],s=100,color=cm.Greens(.33),edgecolors='k',lw=1,zorder=10)
plt.scatter(np.datetime64('2020-07-15'),dAnn['NH-SH'][-3],s=150,color=cm.Greens(.67),edgecolors='k',lw=1,zorder=11)

plt.plot(2*[np.datetime64('2020-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2020-12-31')],[-2,2],'k',ls='dotted',lw=1)

plt.fill_between([np.datetime64('2020-01-01'),np.datetime64('2020-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.01),zorder=0)

plt.text(np.datetime64('2020-07-01'),1.75,r'COV',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2020-07-01'),-1.95,'%.2f' % (dAnn['NH-SH'][-3]),ha='center',va='bottom',fontsize=fs-2)

plt.xlim(fTr.time[0].values,fTr.time[-1].values)
plt.ylim(-2,2)

plt.ylabel(r'$\Delta (S A_\mathrm{aer})^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(fontsize=fs-2)

ax2.text(-.125,1.025,s='b',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

#NH-SH

xvars = ['R','Rsfc','Raer']+['R%s' % cld for cld in cldtypes]
bars_all = [dCov['NH-SH'][var] for var in xvars]

bars_tr = np.array([(dCov['NHtr'][var]-dCov['SHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dCov['NHex'][var]-dCov['SHex'][var]) for var in xvars])/2

ax3 = plt.subplot(5,1,3)

bars = plt.bar(np.arange(len(bars_all)),bars_all,width=.67,color=colors,capsize=4,edgecolor='k',lw=1)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

plt.scatter(np.arange(len(bars_all)),bars_tr,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tropics')
plt.scatter(np.arange(len(bars_all)),bars_ex,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Extratropics')

plt.plot([-1,12],[0,0],'k--',lw=1)

plt.xlim(-.67,11.67)
plt.xticks(np.arange(len(bars_all)),xticks,fontsize=fs)
plt.ylabel(r'COV $\Delta R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-1,.2)

ax3.text(-.125,1.025,s='c',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')

#NH

bars_all = [dCov['NH'][var] for var in xvars]

bars_tr = np.array([(dCov['NHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dCov['NHex'][var]) for var in xvars])/2

ax4 = plt.subplot(5,1,4)

bars = plt.bar(np.arange(len(bars_all)),bars_all,width=.67,color=colors,capsize=4,edgecolor='k',lw=1)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

plt.scatter(np.arange(len(bars_all)),bars_tr,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tropics')
plt.scatter(np.arange(len(bars_all)),bars_ex,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Extratropics')

plt.plot([-1,12],[0,0],'k--',lw=1)

plt.xlim(-.67,11.67)
plt.xticks(np.arange(len(bars_all)),xticks,fontsize=fs)
plt.ylabel(r'COV NH $R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-1,.2)

ax4.text(-.125,1.025,s='d',transform = ax4.transAxes,fontsize=fs+2,fontweight='bold')

#SH

bars_all = [dCov['SH'][var] for var in xvars]

bars_tr = np.array([(dCov['SHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dCov['SHex'][var]) for var in xvars])/2

ax5 = plt.subplot(5,1,5)

bars = plt.bar(np.arange(len(bars_all)),bars_all,width=.67,color=colors,capsize=4,edgecolor='k',lw=1)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

plt.scatter(np.arange(len(bars_all)),bars_tr,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tropics')
plt.scatter(np.arange(len(bars_all)),bars_ex,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Extratropics')

plt.plot([-1,12],[0,0],'k--',lw=1)

plt.legend(frameon=True,fontsize=fs-2,loc=4)

plt.xlim(-.67,11.67)
plt.xticks(np.arange(len(bars_all)),xticks,fontsize=fs)
plt.ylabel(r'COV SH $R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-.6,.6)

ax5.text(-.125,1.025,s='e',transform = ax5.transAxes,fontsize=fs+2,fontweight='bold')


plt.savefig(dir_fig+'Fig4.png',dpi=150)
















