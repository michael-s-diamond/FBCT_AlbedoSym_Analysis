"""
Code to reproduce Figure 1 in Diamond et al. (2024), GRL

Climatology and trends of Earth's hemispheric albedo symmetry by cloud type

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
16 July 2024: Michael Diamond, Tallahassee, FL
    -Merging time-averaging results as panels (c) and (d)
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
fbct['Ra_low'] = fbct['Ra_Cu']+fbct['Ra_Sc']+fbct['Ra_St']
fbct['Ra_mid'] = fbct['Ra_Ac']+fbct['Ra_As']+fbct['Ra_Ns']
fbct['Ra_high'] = fbct['Ra_Ci']+fbct['Ra_Cs']+fbct['Ra_Cb']

fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']
fTr['Ra_low'] = fTr['Ra_cld'][:3].sum(axis=0)
fTr['Ra_mid'] = fTr['Ra_cld'][3:6].sum(axis=0)
fTr['Ra_high'] = fTr['Ra_cld'][6:].sum(axis=0)

"""
Create plot for Figure 1
"""

#
###Panel a: Waterfall plot of climatology
#

Mwts = np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])
climo = []
climo.append(np.average(fTr['Rc'].sel(reg='NH'),weights=Mwts)) #NH average
climo.append(np.average(fTr['Rc_sfc'].sel(reg='NH-SH'),weights=Mwts)) #NH-SH sfc average
climo.append(np.average(fTr['Rc_aer'].sel(reg='NH-SH'),weights=Mwts)) #NH-SH aer average
for ct in cldtypes:#NH-SH atm averages
    climo.append(np.average(fTr['Rc_cld'].sel(cld=ct,reg='NH-SH'),weights=Mwts))
climo.append(np.average(fTr['Rc'].sel(reg='SH'),weights=Mwts)) #SH average

ybase = [0]+[np.sum(climo[:i]) for i in range(1,len(climo)-1)]+[0]

colors0 = ['.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75)]


#
###Panel b: Zonal differences
#

dZ = {}
for var in ['Rc','Rc_sfc','Rc_aer']+['Rc_%s' % ct for ct in cldtypes]:
    dZ[var] = np.average(fbct[var],weights=Mwts[:,np.newaxis,np.newaxis]*np.ones(fbct[var].shape),axis=(0,-1))

colors1 = [cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Reds(.25),cm.Reds(.5),cm.Reds(.75),cm.Purples(.25),cm.Purples(.5),cm.Purples(.75),cm.Blues(.25),cm.Blues(.5),cm.Blues(.75)]


#
###Panels c-d: Calculate averages over different numbers of consecutive months
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

#Get natural experiment values

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
dR[exp], dV[exp], dT[exp] = exp_diff('Ra_low','NH',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
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
###Plot values
#

plt.figure(figsize=(7.5,16))
plt.clf()
fs = 15

#(a) Waterfall plot

ax1 = plt.subplot(3,1,1)

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

plt.text(0,climo[0]+.5,'%.1f' % climo[0],fontsize=fs-2,ha='center',weight='black')
for i in range(1,12):
    if climo[i] > 0: 
        label = '+%.1f' % climo[i]
        offset = .5
    elif climo[i] < 0:
        label = '%.1f' % climo[i]
        offset = -.5
    plt.text(i,climo[i]+ybase[i]+offset,label,fontsize=fs-3,ha='center',va='center')
plt.text(12,climo[-1]+.5,'%.1f' % climo[-1],fontsize=fs-3,ha='center',weight='black')
    
plt.ylim(94,106)

plt.ylabel(r'$\overline{R}$ ($\mathrm{W/m^{2}}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(13),['NH','sfc','aer','Ci', 'Cs', 'Cb','Ac','As','Ns','Cu','Sc','St','SH'],fontsize=fs-2)

#Inset
it = 101.625
ib = 95.625
il = 6-1.35
ir = 6+1.35

plt.plot([il,ir,ir,il,il],[it,it,ib,ib,it],'k',lw=2)
plt.plot([il,ir],[it-2,it-2],'k',lw=2)
plt.plot([il,ir],[it-4,it-4],'k',lw=2)
plt.plot([il+.9,il+.9],[it,ib],'k',lw=2)
plt.plot([ir-.9,ir-.9],[it,ib],'k',lw=2)

plt.text((il+ir)/2,ib-.5,r'$\tau_\mathrm{c}$',fontsize=fs-4,ha='center',va='top')
plt.text(il+.9,ib-.125,r'3.6',fontsize=fs-6,ha='center',va='top')
plt.text(ir-.9,ib-.125,r'23',fontsize=fs-6,ha='center',va='top')

plt.text(il-.5,it-3,'%s\n(hPa)' % r'$p_\mathrm{eff}$',fontsize=fs-4,ha='right',va='center')
plt.text(il-.125,it-2,r'440',fontsize=fs-6,ha='right',va='center')
plt.text(il-.125,it-4,r'680',fontsize=fs-6,ha='right',va='center')

plt.fill_between([il,il+.9],[it,it],[it-2,it-2],facecolor=cm.Reds(.25))
plt.text(il+.45,it-1,'Ci',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([il+.9,ir-.9],[it,it],[it-2,it-2],facecolor=cm.Reds(.5))
plt.text(il+.9+.45,it-1,'Cs',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([ir-.9,ir],[it,it],[it-2,it-2],facecolor=cm.Reds(.75))
plt.text(ir-.45,it-1,'Cb',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([il,il+.9],[it-2,it-2],[it-4,it-4],facecolor=cm.Purples(.25))
plt.text(il+.45,it-3,'Ac',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([il+.9,ir-.9],[it-2,it-2],[it-4,it-4],facecolor=cm.Purples(.5))
plt.text(il+.9+.45,it-3,'As',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([ir-.9,ir],[it-2,it-2],[it-4,it-4],facecolor=cm.Purples(.75))
plt.text(ir-.45,it-3,'Ns',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([il,il+.9],[it-4,it-4],[ib,ib],facecolor=cm.Blues(.25))
plt.text(il+.45,it-5,'Cu',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([il+.9,ir-.9],[it-4,it-4],[ib,ib],facecolor=cm.Blues(.5))
plt.text(il+.9+.45,it-5,'Sc',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

plt.fill_between([ir-.9,ir],[it-4,it-4],[ib,ib],facecolor=cm.Blues(.75))
plt.text(ir-.45,it-5,'St',fontsize=fs-2,ha='center',va='center',color='w',fontweight='bold')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.text(-.15,1,s='(a)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#(b) Zonal differences

ax3 = plt.subplot(6,1,3)

ax3.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),np.array(dZ['Rc'][90:]-dZ['Rc'][:90][::-1]),'k',lw=5,label='Total')
base3 = np.array(dZ['Rc_sfc'][90:]-dZ['Rc_sfc'][:90][::-1]) #Surface
ax3.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),base3,c=colors1[0],zorder=1-i,lw=5,label='sfc',linestyle='dotted')

for i in range(2,len(dZ.keys())):
    
    if i == 2:
        hatch = '/'
        ls = 'dashed'
    else:
        hatch = ''
        ls = 'solid'
    
    ct = list(dZ.keys())[i]
    
    base3 = np.array(dZ[ct][90:]-dZ[ct][:90][::-1])
        
    if i == 2: label = 'aer'
    else: label = ct.split('_')[-1]
    ax3.plot(np.sin(fbct['lat'].values[90:]*np.pi/180),base3,c=colors1[i-1],zorder=1-i,lw=3,label=label,linestyle=ls)
    
ax3.plot(np.sin(fbct['lat'].values*np.pi/180),0*np.sin(fbct['lat'].values*np.pi/180),'k',linestyle='dashed',lw=2,zorder=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.axis([0,1,-40,20])
ax3.tick_params(labelsize=fs-2)
xticks = np.arange(0,91,15)
xticklabs = [r'0$\degree$',r'15$\degree$',r'30$\degree$',r'45$\degree$',r'60$\degree$','',r'90$\degree$']
ax3.set_xticks(np.sin(xticks*np.pi/180))
ax3.set_xticklabels(xticklabs)
ax3.set_ylabel(r'$\Delta \overline{R}$ ($\mathrm{W/m^2}$)',fontsize=fs)
ax3.legend(frameon=False,fontsize=fs-3,ncol=4,loc=3)
ax3.text(-.15,1,s='(b)',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')

#(c-d) Time averages

ax1 = plt.subplot(6,1,4)
plt.plot([0,120],[0,0],'k--',lw=1,zorder=11)
plt.plot(np.arange(1,121),mean,c='k',lw=2,label='Mean',zorder=10)
plt.fill_between(np.arange(1,121),p25,p75,facecolor='.5',label='IQR',zorder=3)
plt.fill_between(np.arange(1,121),p025,p975,facecolor='.75',label='95%',zorder=2)
plt.fill_between(np.arange(1,121),xmin,xmax,facecolor='.95',label='Max-min',zorder=1)
plt.plot(np.arange(1,121),xmax,'k',lw=1,ls='dotted')
plt.plot(np.arange(1,121),xmin,'k',lw=1,ls='dotted')

for exp in dM.keys():
    plt.scatter(dT[exp],dR[exp],s=200,color=dC[exp],marker=dM[exp],lw=1,edgecolors='k',zorder=12)
    
plt.legend(frameon=False,fontsize=fs-3,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)
plt.xlabel('Number of consecutive months averaged',fontsize=fs)

plt.ylim(-4,4)
plt.yticks(fontsize=fs-2)
plt.ylabel(r'$\Delta R^\prime$ ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax1.text(-.15,1,s='(c)',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

ax2 = plt.subplot(6,1,5)

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

plt.legend(frameon=False,fontsize=fs-3,ncol=4)

plt.xlim(1,120)
plt.xticks(np.arange(12,121,12),fontsize=fs-2)
plt.xlabel('Number of consecutive months averaged',fontsize=fs)

plt.yscale('log')
plt.ylim(.01,10)
plt.yticks([.01,.1,1,10],[.01,.1,1,10],fontsize=fs-2)
plt.ylabel(r'|$\Delta R^\prime$| ($\mathrm{W}/\mathrm{m^2}$)',fontsize=fs-2)

ax2.text(-.15,1,s='(d)',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')

plt.tight_layout()

plt.savefig(dir_fig+'Fig1.png',dpi=450)
plt.savefig(dir_fig+'Fig1.pdf')
