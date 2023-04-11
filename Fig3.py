"""
Code to reproduce Figure 3 in Diamond et al. (2023), GRL

"Natural experiment" from Antarctic sea ice decline

Modification history
--------------------
05 April 2023: Michael Diamond, Tallahassee, FL
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

"""
Analyze pre- and post-Antarctic sea ice decline periods
"""

pre = np.logical_and(fTr.time.dt.year>=2012,fTr.time.dt.year<2016)
post = np.logical_and(fTr.time.dt.year>=2016,fTr.time.dt.year<2020)

dPre = {reg : {} for reg in fTr.reg.values}
dPos = {reg : {} for reg in fTr.reg.values}

for reg in fTr.reg.values:
    
    #Clear-sky
    dPre[reg]['SAclr'] = np.average(fTr['S*Aa_clr'].sel(reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['SAclr'] = np.average(fTr['S*Aa_clr'].sel(reg=reg)[post],weights=Mwts[post])
    
    dPre[reg]['SAsfc'] = np.average(fTr['S*Aa_sfc'].sel(reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['SAsfc'] = np.average(fTr['S*Aa_sfc'].sel(reg=reg)[post],weights=Mwts[post])
    
    dPre[reg]['SAaer'] = np.average(fTr['S*Aa_atm'].sel(cld='clr',reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['SAaer'] = np.average(fTr['S*Aa_atm'].sel(cld='clr',reg=reg)[post],weights=Mwts[post])
    
    #All-sky
    dPre[reg]['R'] = np.average(fTr['Ra'].sel(reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['R'] = np.average(fTr['Ra'].sel(reg=reg)[post],weights=Mwts[post])
    
    dPre[reg]['Rsfc'] = np.average(fTr['Ra_sfc'].sel(reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['Rsfc'] = np.average(fTr['Ra_sfc'].sel(reg=reg)[post],weights=Mwts[post])
    
    dPre[reg]['Raer'] = np.average(fTr['Ra_atm'].sel(cld='clr',reg=reg)[pre],weights=Mwts[pre])
    dPos[reg]['Raer'] = np.average(fTr['Ra_atm'].sel(cld='clr',reg=reg)[post],weights=Mwts[post])
    
    for cld in cldtypes:
        dPre[reg]['R%s' % cld] = np.average(fTr['Ra_atm'].sel(cld=cld,reg=reg)[pre],weights=Mwts[pre])
        dPos[reg]['R%s' % cld] = np.average(fTr['Ra_atm'].sel(cld=cld,reg=reg)[post],weights=Mwts[post])

        
        
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

#SH SAsfc

ax1 = plt.subplot(5,1,1)

plt.plot(fTr.time,fTr['S*Aa_sfc'].sel(reg='SH'),c='k',lw=3)
plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=1)

plt.plot([np.datetime64('2012-01-01'),np.datetime64('2015-12-31')],2*[dPre['SH']['SAsfc']],c=cm.Greens(.33),lw=5,solid_capstyle='butt')
plt.plot([np.datetime64('2016-01-01'),np.datetime64('2019-12-31')],2*[dPos['SH']['SAsfc']],c=cm.Greens(.67),lw=5,solid_capstyle='butt')

plt.plot(2*[np.datetime64('2012-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2016-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2020-01-01')],[-2,2],'k',ls='dotted',lw=1)

plt.fill_between([np.datetime64('2012-01-01'),np.datetime64('2015-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.01),zorder=0)
plt.fill_between([np.datetime64('2016-01-01'),np.datetime64('2019-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.25),zorder=0)

plt.text(np.datetime64('2014-01-01'),1.75,r'Pre',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2018-01-01'),1.75,'Post',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2014-01-01'),-1.95,r'%.2f $\mathrm{W/m^2}$' % (dPre['SH']['SAsfc']),ha='center',va='bottom',fontsize=fs-2)
plt.text(np.datetime64('2018-01-01'),-1.95,'%.2f $\mathrm{W/m^2}$' % (dPos['SH']['SAsfc']),ha='center',va='bottom',fontsize=fs-2)

plt.xlim(fTr.time[0].values,fTr.time[-1].values)
plt.ylim(-2,2)

plt.ylabel(r'SH $(S A_\mathrm{sfc})^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(fontsize=fs-2)

ax1.text(-.125,1.025,s='a',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#NH-SH SAsfc

ax2 = plt.subplot(5,1,2)

plt.plot(fTr.time,fTr['S*Aa_sfc'].sel(reg='NH-SH'),c='k',lw=3)
plt.plot(fTr.time,np.zeros(len(fTr.time)),'k--',lw=1)

plt.plot([np.datetime64('2012-01-01'),np.datetime64('2015-12-31')],2*[dPre['NH-SH']['SAsfc']],c=cm.Greens(.33),lw=5,solid_capstyle='butt')
plt.plot([np.datetime64('2016-01-01'),np.datetime64('2019-12-31')],2*[dPos['NH-SH']['SAsfc']],c=cm.Greens(.67),lw=5,solid_capstyle='butt')

plt.plot(2*[np.datetime64('2012-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2016-01-01')],[-2,2],'k',ls='dotted',lw=1)
plt.plot(2*[np.datetime64('2020-01-01')],[-2,2],'k',ls='dotted',lw=1)

plt.fill_between([np.datetime64('2012-01-01'),np.datetime64('2015-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.01),zorder=0)
plt.fill_between([np.datetime64('2016-01-01'),np.datetime64('2019-12-31')],[-2,-2],[2,2],facecolor=cm.Greens(.25),zorder=0)

plt.text(np.datetime64('2014-01-01'),1.75,r'Pre',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2018-01-01'),1.75,'Post',ha='center',va='top',fontsize=fs-2)
plt.text(np.datetime64('2014-01-01'),-1.95,r'%.2f $\mathrm{W/m^2}$' % (dPre['NH-SH']['SAsfc']),ha='center',va='bottom',fontsize=fs-2)
plt.text(np.datetime64('2018-01-01'),-1.95,'%.2f $\mathrm{W/m^2}$' % (dPos['NH-SH']['SAsfc']),ha='center',va='bottom',fontsize=fs-2)

plt.xlim(fTr.time[0].values,fTr.time[-1].values)
plt.ylim(-2,2)

plt.ylabel(r'$\Delta (S A_\mathrm{sfc})^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(fontsize=fs-2)

ax2.text(-.125,1.025,s='b',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

#NH-SH

xvars = ['R','Rsfc','Raer']+['R%s' % cld for cld in cldtypes]
bars_all = [dPos['NH-SH'][var]-dPre['NH-SH'][var] for var in xvars]

bars_tr = np.array([(dPos['NHtr'][var]-dPos['SHtr'][var])-(dPre['NHtr'][var]-dPre['SHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dPos['NHex'][var]-dPos['SHex'][var])-(dPre['NHex'][var]-dPre['SHex'][var]) for var in xvars])/2

ax3 = plt.subplot(5,1,3)

bars = plt.bar(np.arange(len(bars_all)),bars_all,width=.67,color=colors,capsize=4,edgecolor='k',lw=1)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

plt.scatter(np.arange(len(bars_all)),bars_tr,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tropics')
plt.scatter(np.arange(len(bars_all)),bars_ex,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Extratropics')

plt.plot([-1,12],[0,0],'k--',lw=1)

plt.xlim(-.67,11.67)
plt.xticks(np.arange(len(bars_all)),xticks,fontsize=fs)
plt.ylabel(r'Post-Pre $\Delta R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-.6,.6)

ax3.text(-.125,1.025,s='c',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')

#NH

bars_all = [dPos['NH'][var]-dPre['NH'][var] for var in xvars]

bars_tr = np.array([(dPos['NHtr'][var])-(dPre['NHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dPos['NHex'][var])-(dPre['NHex'][var]) for var in xvars])/2

ax4 = plt.subplot(5,1,4)

bars = plt.bar(np.arange(len(bars_all)),bars_all,width=.67,color=colors,capsize=4,edgecolor='k',lw=1)

for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
    bar.set_hatch(hatch)

plt.scatter(np.arange(len(bars_all)),bars_tr,s=100,facecolor='k',edgecolors='w',lw=1,marker='P',label='Tropics')
plt.scatter(np.arange(len(bars_all)),bars_ex,s=100,facecolor='k',edgecolors='w',lw=1,marker='X',label='Extratropics')

plt.plot([-1,12],[0,0],'k--',lw=1)

plt.xlim(-.67,11.67)
plt.xticks(np.arange(len(bars_all)),xticks,fontsize=fs)
plt.ylabel(r'Post-Pre NH $R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-.8,.4)

ax4.text(-.125,1.025,s='d',transform = ax4.transAxes,fontsize=fs+2,fontweight='bold')

#SH

bars_all = [dPos['SH'][var]-dPre['SH'][var] for var in xvars]

bars_tr = np.array([(dPos['SHtr'][var])-(dPre['SHtr'][var]) for var in xvars])/2
bars_ex = np.array([(dPos['SHex'][var])-(dPre['SHex'][var]) for var in xvars])/2

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
plt.ylabel(r'Post-Pre SH $R^\prime$ ($\mathrm{W/m^2}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.ylim(-.8,.4)

ax5.text(-.125,1.025,s='e',transform = ax5.transAxes,fontsize=fs+2,fontweight='bold')


plt.savefig(dir_fig+'Fig3.png',dpi=150)
















