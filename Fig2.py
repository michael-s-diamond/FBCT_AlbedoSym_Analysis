"""
Code to reproduce Figure 2 in Diamond et al. (2023), GRL

Lagged correlations and trends of clear-sky and cloudy components

Modification history
--------------------
30 March 2023: Michael Diamond, Tallahassee, FL
    -Created
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
fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']


"""
4-year rolling (top-hat) averaging of time series
"""

N = 4*12
high = np.logical_or(fTr.cld.values=='Ci',np.logical_or(fTr.cld.values=='Cs',fTr.cld.values=='Cb'))
mid = np.logical_or(fTr.cld.values=='Ac',np.logical_or(fTr.cld.values=='As',fTr.cld.values=='Ns'))
low = np.logical_or(fTr.cld.values=='Cu',np.logical_or(fTr.cld.values=='Sc',fTr.cld.values=='St'))

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fTr.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

#Global all-sky NH-SH
Dall = np.array([np.average(fTr['Ra'].sel(reg='NH-SH')[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

#Global clear-sky NH-SH
Dclr = np.array([np.average(fTr['S*Aa_clr'].sel(reg='NH-SH')[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

Dsfc = np.array([np.average(fTr['Ra_sfc'].sel(reg='NH-SH')[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])
Daer = np.array([np.average(fTr['Ra_atm'].sel(cld='clr',reg='NH-SH')[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])


#Tropical high cloud NH-SH
NHtrh = fTr['Ra_atm'][high].sel(reg='NHtr').sum(axis=0).values
SHtrh = fTr['Ra_atm'][high].sel(reg='SHtr').sum(axis=0).values
Dtrh = np.array([np.average((NHtrh-SHtrh)[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

#Tropical low+mid cloud NH-SH
NHtrl = fTr['Ra_atm'][np.logical_or(low,mid)].sel(reg='NHtr').sum(axis=0).values
SHtrl = fTr['Ra_atm'][np.logical_or(low,mid)].sel(reg='SHtr').sum(axis=0).values
Dtrl = np.array([np.average((NHtrl-SHtrl)[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

#Extratropical low+mid cloud NH-SH
NHexl = fTr['Ra_atm'][np.logical_or(low,mid)].sel(reg='NHex').sum(axis=0).values
SHexl = fTr['Ra_atm'][np.logical_or(low,mid)].sel(reg='SHex').sum(axis=0).values
Dexl = np.array([np.average((NHexl-SHexl)[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

#Extratropical high cloud NH-SH
NHexh = fTr['Ra_atm'][high].sel(reg='NHex').sum(axis=0).values
SHexh = fTr['Ra_atm'][high].sel(reg='SHex').sum(axis=0).values
Dexh = np.array([np.average((NHexh-SHexh)[i:N+i],weights=Mwts[i:N+i]) for i in range(len(fTr.time)-N)])

#Lagged correlations
rlag_trh = [stats.pearsonr(Dtrh,Dclr)[0]]
rlag_exl = [stats.pearsonr(Dexl,Dclr)[0]]
rlag_trl = [stats.pearsonr(Dtrl,Dclr)[0]]
rlag_exh = [stats.pearsonr(Dexh,Dclr)[0]]
rlag_tr = [stats.pearsonr(Dtrh+Dtrl,Dclr)[0]]
rlag_ex = [stats.pearsonr(Dexh+Dexl,Dclr)[0]]

for i in range(1,97):
    rlag_trh.append(stats.pearsonr(Dtrh[i:],Dclr[:-i])[0])
    rlag_exl.append(stats.pearsonr(Dexl[i:],Dclr[:-i])[0])
    rlag_trl.append(stats.pearsonr(Dtrl[i:],Dclr[:-i])[0])
    rlag_exh.append(stats.pearsonr(Dexh[i:],Dclr[:-i])[0])
    rlag_tr.append(stats.pearsonr((Dtrh+Dtrl)[i:],Dclr[:-i])[0])
    rlag_ex.append(stats.pearsonr((Dexh+Dexl)[i:],Dclr[:-i])[0])

        
"""
Decadal trends
"""    

dTr = {} #Tropical
dEx = {} #Extratropical

for cld in cldtypes:
    
    dTr[cld] = fTr['Ra_atm'].sel(reg='NHtr',cld=cld)-fTr['Ra_atm'].sel(reg='SHtr',cld=cld)
    dEx[cld] = fTr['Ra_atm'].sel(reg='NHex',cld=cld)-fTr['Ra_atm'].sel(reg='SHex',cld=cld)
    
    #Trends for tropics and extratropics (1/2 weighting for global effect)
    dTr[cld+'_trend'] = (fTr['Ra_atm_trend'].sel(reg='NHtr',cld=cld)-fTr['Ra_atm_trend'].sel(reg='SHtr',cld=cld)).values/2
    dEx[cld+'_trend'] = (fTr['Ra_atm_trend'].sel(reg='NHex',cld=cld)-fTr['Ra_atm_trend'].sel(reg='SHex',cld=cld)).values/2
    
    dTr[cld+'_trend_err'] = np.sqrt((fTr['Ra_atm_trend_err'].sel(reg='NHtr',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_atm_nu'].sel(reg='NHtr',cld=cld)))**2+(fTr['Ra_atm_trend_err'].sel(reg='SHtr',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_atm_nu'].sel(reg='SHtr',cld=cld)))**2).values
    dEx[cld+'_trend_err'] = np.sqrt((fTr['Ra_atm_trend_err'].sel(reg='NHex',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_atm_nu'].sel(reg='NHex',cld=cld)))**2+(fTr['Ra_atm_trend_err'].sel(reg='SHex',cld=cld)/2*stats.t.ppf(.975,fTr['Ra_atm_nu'].sel(reg='SHex',cld=cld)))**2).values

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

dTr['low+mid'] = dTr['low']+dTr['mid']
dTr['low+mid_trend'] = dTr['low_trend']+dTr['mid_trend']
dTr['low+mid_trend_err'] = np.sqrt(dTr['low_trend_err']**2+dTr['mid_trend_err']**2)

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

dEx['low+mid'] = dEx['low']+dEx['mid']
dEx['low+mid_trend'] = dEx['low_trend']+dEx['mid_trend']
dEx['low+mid_trend_err'] = np.sqrt(dEx['low_trend_err']**2+dEx['mid_trend_err']**2)


#Clear-sky 

dClr = {} #Clear-sky
Cclr = np.average(fTr['Cc'].sel(reg='global',cld='clr'),weights=np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])) #Weight by average clear-sky fraction (1-C)

dClr['all_trend'] = Cclr*fTr['S*Aa_clr_trend'].sel(reg='NH-SH').values #Both surface and atmosphere
dClr['all_trend_err'] = (Cclr*fTr['S*Aa_clr_trend_err'].sel(reg='NH-SH')*stats.t.ppf(.975,fTr['S*Aa_clr_nu'].sel(reg='NH-SH'))).values

dClr['aer_trend'] = Cclr*fTr['S*Aa_atm_trend'].sel(cld='clr',reg='NH-SH').values 
dClr['aer_trend_err'] = (Cclr*fTr['S*Aa_atm_trend_err'].sel(cld='clr',reg='NH-SH')*stats.t.ppf(.975,fTr['S*Aa_atm_nu'].sel(cld='clr',reg='NH-SH'))).values 

dClr['sfc_trend'] = Cclr*fTr['S*Aa_sfc_trend'].sel(reg='NH-SH').values
dClr['sfc_trend_err'] = (Cclr*fTr['S*Aa_sfc_trend_err'].sel(reg='NH-SH')*stats.t.ppf(.975,fTr['S*Aa_sfc_nu'].sel(reg='NH-SH'))).values



"""
Plot
"""

dColors = {'Cu' : cm.Blues(.25), 'Sc' : cm.Blues(.5), 'St' : cm.Blues(.75), 'Ac' : cm.Purples(.25), 'As' : cm.Purples(.5), 'Ns' : cm.Purples(.75), 'Ci' : cm.Reds(.25), 'Cs' : cm.Reds(.5), 'Cb' : cm.Reds(.75)}
dMarkers = {'Cu' : 'v', 'Sc' : 'v', 'St' : 'v', 'Ac' : 's', 'As' : 's', 'Ns' : 's', 'Ci' : '^', 'Cs' : '^', 'Cb' : '^'}


#Set up plot
plt.figure(figsize=(12,14))
plt.clf()
fs = 15

#
### (a) Lagged correlations
#

ax1 = plt.subplot(3,1,1)

plt.plot(rlag_trh,c=cm.Reds(.67),ls='solid',lw=3,label=r'tropical $\Delta R^\prime_\mathrm{high}$')
plt.plot(rlag_trl,c=cm.Blues(.33),ls='solid',lw=3,label=r'tropical $\Delta R^\prime_\mathrm{low+mid}$')
plt.plot(rlag_exh,c=cm.Reds(.67),ls='dashed',lw=3,label=r'extratropical $\Delta R^\prime_\mathrm{high}$')
plt.plot(rlag_exl,c=cm.Blues(.33),ls='dashed',lw=3,label=r'extratropical $\Delta R^\prime_\mathrm{low+mid}$')

plt.plot([-1,121],[0,0],'k--',lw=1,zorder=0)

for r2 in np.arange(-1,0,.1):
    plt.plot([-1,121],2*[-np.sqrt(np.abs(r2))],'k',linestyle='dotted',lw=1)

plt.legend(fontsize=fs-3,frameon=False,loc=9,ncol=4)

plt.ylim(-1,1)
plt.xlim(0,96)

plt.ylabel(r"Pearson's $r$",fontsize=fs)
plt.yticks(np.arange(-1,1.1,.25),fontsize=fs-2)
plt.xlabel('Lag (months)',fontsize=fs)
plt.xticks(np.arange(0,97,12),fontsize=fs-2)

plt.title('Lagged correlations between 4-year rolling averages of $\Delta(SA_\mathrm{clr})^\prime$ and...',fontsize=fs)

ax1.text(-.125,1.025,s='a',transform = ax1.transAxes,fontsize=fs+2,fontweight='bold')

#
### (b) Tropical trends
#

ax2 = plt.subplot(3,1,2)

plt.plot([-1,11],[0,0],'k--',lw=1)

#CLR

plt.plot([-.5,1.5],2*[dClr['all_trend']],lw=3,c='.05',solid_capstyle='butt')
plt.fill_between([-.5,1.5],2*[dClr['all_trend']-dClr['all_trend_err']],2*[dClr['all_trend']+dClr['all_trend_err']],facecolor='.95')

plt.scatter(0,dClr['sfc_trend'],s=250,color=cm.YlOrBr(.33),marker='*',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(0,dClr['sfc_trend'],yerr=dClr['sfc_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.scatter(1,dClr['aer_trend'],s=250,color=cm.YlOrBr(.67),marker='*',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(1,dClr['aer_trend'],yerr=dClr['aer_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(.5,.45,'Clear-sky\n%s' % r'(1-$\overline{C}$) d$(SA_\mathrm{clr})^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([1.5,1.5],[-1,1],'k',lw=1,linestyle='solid')

#Tropics

plt.plot([1.5,4.5],2*[dTr['high_trend']],lw=3,color=cm.Reds(.95),solid_capstyle='butt')
plt.fill_between([1.5,4.5],2*[dTr['high_trend']-dTr['high_trend_err']],2*[dTr['high_trend']+dTr['high_trend_err']],facecolor=cm.Reds(.05))

n = 1
for cld in ['Ci','Cs','Cb']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(3,.45,'Tropical high clouds\n%s' % r'0.5 d$R_\mathrm{high}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([4.5,4.5],[-1,1],'k',lw=1,linestyle='solid')
    
#SO mid

plt.plot([4.5,7.5],2*[dTr['mid_trend']],lw=3,color=cm.Purples(.95),solid_capstyle='butt')
plt.fill_between([4.5,7.5],2*[dTr['mid_trend']-dTr['mid_trend_err']],2*[dTr['mid_trend']+dTr['mid_trend_err']],facecolor=cm.Purples(.05))

for cld in ['Ac','As','Ns']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(6,.45,'Tropical mid clouds\n%s' % r'0.5 d$R_\mathrm{mid}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([7.5,7.5],[-1,1],'k',lw=1,linestyle='dotted')

#SO low

plt.plot([7.5,10.5],2*[dTr['low_trend']],lw=3,color=cm.Blues(.95),solid_capstyle='butt')
plt.fill_between([7.5,10.5],2*[dTr['low_trend']-dTr['low_trend_err']],2*[dTr['low_trend']+dTr['low_trend_err']],facecolor=cm.Blues(.05))

for cld in ['Cu','Sc','St']:
    n += 1
    plt.scatter(n,dTr[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dTr[cld+'_trend'],yerr=dTr[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(9,.45,'Tropical low clouds\n%s' % r'0.5 d$R_\mathrm{low}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)

plt.xlim(-.5,10.5)
plt.ylim(-.5,.5)

plt.ylabel(r'NH-SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(11),['sfc','aer','Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St'],fontsize=fs)

ax2.text(-.125,1.025,s='b',transform = ax2.transAxes,fontsize=fs+2,fontweight='bold')


#
### (c) Extratropical trends
#

ax3 = plt.subplot(3,1,3)

plt.plot([-1,11],[0,0],'k--',lw=1)

#CLR

plt.plot([-.5,1.5],2*[dClr['all_trend']],lw=3,c='.05',solid_capstyle='butt')
plt.fill_between([-.5,1.5],2*[dClr['all_trend']-dClr['all_trend_err']],2*[dClr['all_trend']+dClr['all_trend_err']],facecolor='.95')

plt.scatter(0,dClr['sfc_trend'],s=250,color=cm.YlOrBr(.33),marker='*',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(0,dClr['sfc_trend'],yerr=dClr['sfc_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.scatter(1,dClr['aer_trend'],s=250,color=cm.YlOrBr(.67),marker='*',linewidth=1,edgecolors='k',zorder=7)
plt.errorbar(1,dClr['aer_trend'],yerr=dClr['aer_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(.5,.45,'Clear-sky\n%s' % r'(1-$\overline{C}$) d$(SA_\mathrm{clr})^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)

plt.plot([1.5,1.5],[-1,1],'k',lw=1,linestyle='solid')

#Tropics

plt.plot([1.5,4.5],2*[dEx['high_trend']],lw=3,color=cm.Reds(.95),solid_capstyle='butt')
plt.fill_between([1.5,4.5],2*[dEx['high_trend']-dEx['high_trend_err']],2*[dEx['high_trend']+dEx['high_trend_err']],facecolor=cm.Reds(.05))

n = 1
for cld in ['Ci','Cs','Cb']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(3,.45,'Extratropical high clouds\n%s' % r'0.5 d$R_\mathrm{high}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([4.5,4.5],[-1,1],'k',lw=1,linestyle='solid')
    
#SO mid

plt.plot([4.5,7.5],2*[dEx['mid_trend']],lw=3,color=cm.Purples(.95),solid_capstyle='butt')
plt.fill_between([4.5,7.5],2*[dEx['mid_trend']-dEx['mid_trend_err']],2*[dEx['mid_trend']+dEx['mid_trend_err']],facecolor=cm.Purples(.05))

for cld in ['Ac','As','Ns']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(6,.45,'Extratropical mid clouds\n%s' % r'0.5 d$R_\mathrm{mid}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)
    
plt.plot([7.5,7.5],[-1,1],'k',lw=1,linestyle='dotted')

#SO low

plt.plot([7.5,10.5],2*[dEx['low_trend']],lw=3,color=cm.Blues(.95),solid_capstyle='butt')
plt.fill_between([7.5,10.5],2*[dEx['low_trend']-dEx['low_trend_err']],2*[dEx['low_trend']+dEx['low_trend_err']],facecolor=cm.Blues(.05))

for cld in ['Cu','Sc','St']:
    n += 1
    plt.scatter(n,dEx[cld+'_trend'],s=250,color=dColors[cld],marker=dMarkers[cld],linewidth=1,edgecolors='k',zorder=7)
    plt.errorbar(n,dEx[cld+'_trend'],yerr=dEx[cld+'_trend_err'],color='k',capsize=4,lw=1,zorder=5,fmt='none')

plt.text(9,.45,'Extratropical low clouds\n%s' % r'0.5 d$R_\mathrm{low}^\prime$/d$t$',ha='center',va='top',fontsize=fs-2)

plt.xlim(-.5,10.5)
plt.ylim(-.5,.5)

plt.ylabel(r'NH-SH trend ($\mathrm{W/m^2/decade}$)',fontsize=fs)
plt.yticks(fontsize=fs-2)
plt.xticks(np.arange(11),['sfc','aer','Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St'],fontsize=fs)

ax3.text(-.125,1.025,s='c',transform = ax3.transAxes,fontsize=fs+2,fontweight='bold')


plt.savefig(dir_fig+'Fig2.png',dpi=150)
plt.savefig(dir_fig+'Fig2.eps')
