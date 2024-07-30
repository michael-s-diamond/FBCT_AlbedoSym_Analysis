"""
Code to reproduce Figure 3 in Diamond et al. (2024), ESSOAr

Natural experiments analysis

Modification history
--------------------
24 June 2024: Michael Diamond, Tallahassee, FL
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

fTr = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc')
cldtypes = ['Ci','Cs','Cb','Ac','As','Ns','Cu','Sc','St']
fTr['Ra_low'] = fTr['Ra_cld'][:3].sum(axis=0)
fTr['Ra_mid'] = fTr['Ra_cld'][3:6].sum(axis=0)
fTr['Ra_high'] = fTr['Ra_cld'][6:].sum(axis=0)

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fTr.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

"""
Panel a: Zonal
"""

dZ = {} #Zonal trend or difference
dZe = {} #Zonal trend or difference error (95% confidence)

dt0 = {'pre' : {}, 'post' : {}}
dt1 = {'pre' : {}, 'post' : {}}

#
###Zonal trends for ARC and CHN
#

def Ztrend(var,t0,t1):
    y = fbct[var].sel(time=slice(t0,t1)).mean(axis=-1)
    x = np.arange(len(y))/12/10

    ybar = y.mean(axis=0)
    xbar = x.mean(axis=0)

    beta_hat = np.sum((x-xbar)[:,np.newaxis]*(y-ybar),axis=0)/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar

    yhat = alpha_hat.values[np.newaxis,:]+beta_hat.values[np.newaxis,:]*x[:,np.newaxis]
    e = y-yhat

    xx = e[1:].values
    yy = e[:-1].values
    xm = xx.mean(axis=0)
    ym = yy.mean(axis=0)

    r1e_num = np.sum((xx-xm)*(yy-ym),axis=0)
    r1e_denom = np.sum((xx-xm)**2,axis=0)*np.sum((yy-ym)**2,axis=0)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = np.array(len(x)*(1-r1e)/(1+r1e))
    nue[nue>len(x)] = len(x)

    sbh = np.sqrt(np.sum(e**2,axis=0)/(nue-2)/np.sum((x-xbar)**2))
    
    return beta_hat, sbh

#China post-2012 aerosol decrease
trend, err = Ztrend('Ra_aer',np.datetime64('2010-06-01'),np.datetime64('2019-05-31'))
dZ['CHN'] = trend.values
dZe['CHN'] = err.values

#Arctic sea ice loss 2002-2012
trend, err = Ztrend('Ra_sfc',np.datetime64('2002-07-01'),np.datetime64('2012-06-30'))
dZ['ARC'] = trend.values
dZe['ARC'] = err.values

#
###Zonal pre-post differences for other natural experiments
#

def Zdiff(var,pre_t0,pre_t1,post_t0,post_t1):
    pre_ = fbct[var].sel(time=slice(pre_t0,pre_t1)).mean(axis=-1)
    pre = pre_.mean(axis=0)
    pre_e = fbct[var].mean(axis=-1).rolling(time=pre_.shape[0]).mean().std(axis=0)

    post = fbct[var].sel(time=slice(post_t0,post_t1)).mean(axis=(0,-1))
    
    return pre, post, pre_e

#Antarctic sea ice
exp = 'ANT'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-06-01'),np.datetime64('2015-06-01'),np.datetime64('2015-06-01'),np.datetime64('2019-06-01')
pre, post, e = Zdiff('Ra_sfc',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

#IMO 2020
exp = 'IMO'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2017-07-01'),np.datetime64('2020-01-01'),np.datetime64('2020-01-01'),np.datetime64('2022-07-01')
pre, post, e = Zdiff('Ra_low',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

#COVID-19
exp = 'COV'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-02-01'),np.datetime64('2020-02-01'),np.datetime64('2020-02-01'),np.datetime64('2021-02-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

pre, post, e = Zdiff('Ra_low',np.datetime64('2019-02-01'),np.datetime64('2020-02-01'),np.datetime64('2020-02-01'),np.datetime64('2021-02-01'))
dZ['COV-low'] = (post-pre).values
dZe['COV-low'] = e.values

#Australian wildfires
exp = 'AUS'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-09-01'),np.datetime64('2019-12-01'),np.datetime64('2019-12-01'),np.datetime64('2020-03-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

pre, post, e = Zdiff('Ra_low',np.datetime64('2019-09-01'),np.datetime64('2019-12-01'),np.datetime64('2019-12-01'),np.datetime64('2020-03-01'))
dZ['AUS-low'] = (post-pre).values
dZe['AUS-low'] = e.values

#Raikoke
exp = 'RAI'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2019-04-01'),np.datetime64('2019-06-01'),np.datetime64('2019-07-01'),np.datetime64('2019-09-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

#Nabro
exp = 'NAB'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2011-04-01'),np.datetime64('2011-06-01'),np.datetime64('2011-06-01'),np.datetime64('2011-08-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values


"""#Sarychev
exp = 'SAR'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2009-04-01'),np.datetime64('2009-06-01'),np.datetime64('2009-06-01'),np.datetime64('2009-08-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values


#Kasatochi
exp = 'KAS'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2008-06-01'),np.datetime64('2008-08-01'),np.datetime64('2008-08-01'),np.datetime64('2008-10-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values

#Hunga Tonga
exp = 'HUN'
dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp] = np.datetime64('2021-10-01'),np.datetime64('2022-01-01'),np.datetime64('2022-01-01'),np.datetime64('2022-04-01')
pre, post, e = Zdiff('Ra_aer',dt0['pre'][exp], dt1['pre'][exp], dt0['post'][exp], dt1['post'][exp])
dZ[exp] = (post-pre).values
dZe[exp] = e.values"""



"""
#Testing if weighting by days per month matters (seems to be very small compared to zonal means)

t0 = np.datetime64('2011-06-01')
t1 = np.datetime64('2015-06-01')
t2 = np.datetime64('2019-06-01')

t_pre = np.logical_and(fbct.time>t0,fbct.time<t1)

months = np.array([np.datetime64(t0,'M')+np.timedelta64(i,'M') for i in range(len(fbct.time[t_pre]))])
Mwts_pre = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

pre = np.average(fbct['Ra_sfc'][t_pre],weights=Mwts_pre,axis=0).mean(axis=-1)

t_post = np.logical_and(fbct.time>t1,fbct.time<t2)

months = np.array([np.datetime64(t1,'M')+np.timedelta64(i,'M') for i in range(len(fbct.time[t_post]))])
Mwts_post = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

post = np.average(fbct['Ra_sfc'][t_post],weights=Mwts_post,axis=0).mean(axis=-1)

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(12*(2022-2002))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)
"""



"""
Panel b: Bars
"""

#Dictionaries to store values
dM = {} #Means
dE = {} #Errors
dC = {} #Confidence (95%)

#
###Trends for CHN and ARC
#

def calc_trend(data):
    
    y = data.values
    x = np.arange(len(y))/12/10
    
    ybar = y.mean()
    xbar = x.mean()
    
    beta_hat = np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar
    
    yhat = alpha_hat+beta_hat*x
    e = y-yhat
    
    xx = e[1:]
    yy = e[:-1]
    xm = xx.mean()
    ym = yy.mean()
    
    r1e_num = np.sum((xx-xm)*(yy-ym))
    r1e_denom = np.sum((xx-xm)**2)*np.sum((yy-ym)**2)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = len(y)*(1-r1e)/(1+r1e)
    if nue>len(y): nue = len(y)
    
    sbh = np.sqrt(np.sum(e**2)/(nue-2)/np.sum((x-xbar)**2))
    
    C95 = sbh*stats.t.ppf(.975,nue)
    
    return beta_hat, sbh, C95

for exp, t0, t1 in zip(['ARC','CHN'],[np.datetime64('2002-07-01'),np.datetime64('2010-06-01')],[np.datetime64('2012-06-30'),np.datetime64('2019-05-31')]):
    
    dM[exp] = {}
    dE[exp] = {}
    dC[exp] = {}
    
    for reg in fTr.reg.values:

        y = fTr['Ra'].sel(reg=reg).sel(time=slice(t0,t1))

        m, e, c = calc_trend(y)

        M = [m]
        E = [e]
        C = [c]

        for key in ['sfc','aer','low','mid','high']:

            y = fTr['Ra_'+key].sel(reg=reg).sel(time=slice(t0,t1))

            m, e, c = calc_trend(y)

            M.append(m)
            E.append(e)
            C.append(c)

        dM[exp][reg] = np.array(M)
        dE[exp][reg] = np.array(E)
        dC[exp][reg] = np.array(C)


#
###Differences for other experiments
#

exp_list = ['ANT','IMO','COV','AUS','NAB','RAI']

for exp in exp_list:
    
    dM[exp] = {}
    dE[exp] = {}
    
    for reg in fTr.reg.values:

        pre = fTr['Ra'].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])).mean().values
        post = fTr['Ra'].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])).mean().values
        
        m = post-pre
        e = fTr['Ra'].sel(reg=reg).rolling(time=len(fTr['Ra'].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])))).mean().std(axis=0)

        M = [m]
        E = [e]

        for key in ['sfc','aer','low','mid','high']:

            pre = fTr['Ra_'+key].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])).mean().values
            post = fTr['Ra_'+key].sel(reg=reg).sel(time=slice(dt0['post'][exp],dt1['post'][exp])).mean().values

            m = post-pre
            e = fTr['Ra_'+key].sel(reg=reg).rolling(time=len(fTr['Ra_'+key].sel(reg=reg).sel(time=slice(dt0['pre'][exp],dt1['pre'][exp])))).mean().std(axis=0)

            M.append(m)
            E.append(e)

        dM[exp][reg] = np.array(M)
        dE[exp][reg] = np.array(E)

    
    
"""
Make figure
""" 

#Aesthetics and labels
markers = ['D','*','o','o','^','s','v']
labels = [r'$R^\prime$',r'$R^\prime_{\mathrm{sfc}}$',r'$R^\prime_{\mathrm{aer}}$',r'$R^\prime_{\mathrm{low}}$',r'$R^\prime_{\mathrm{mid}}$',r'$R^\prime_{\mathrm{high}}$']
dlc = {'ARC' : cm.YlOrBr(.33),'CHN' : cm.YlOrBr(.67),'ANT' : cm.YlOrBr(.33),'IMO' : cm.Blues(.5),'COV' : cm.YlOrBr(.67),'AUS' : cm.YlOrBr(.67),'NAB' : cm.YlOrBr(.67),'RAI' : cm.YlOrBr(.67)}
bcolors = ['.5',cm.YlOrBr(.33),cm.YlOrBr(.67),cm.Blues(.25),cm.Purples(.25),cm.Reds(.25)]
bedges = 3*['k']+[cm.Blues(.75),cm.Purples(.75),cm.Reds(.75)]
ecolors = 3*['k']+[cm.Blues(.5),cm.Purples(.5),cm.Reds(.5)]
lab = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
ylab = [r'$R^\prime_{\mathrm{sfc}}$',r'$R^\prime_{\mathrm{aer}}$',r'$R^\prime_{\mathrm{sfc}}$',r'$R^\prime_{\mathrm{low}}$']+4*[r'$R^\prime_{\mathrm{aer}}$']
text = ['Jul 2002 to Jun 2012','Jun 2010 to May 2019','Jun 2015-May 2019 minus Jun 2011-May 2015','Jan 2020-Jun 2022 minus Jul 2017-Dec 2019','Feb 2020-Jan 2021 minus Feb 2019-Jan 2020','Dec 2019-Feb 2020 minus Sep 2019-Nov 2019','Jun 2011-Jul 2011 minus Apr 2011-May 2011','Jul 2019-Aug 2019 minus Apr 2019-May 2019']

#
###Plot
#
plt.figure(figsize=(9,12))
plt.clf()

fs = 14

n = 0
for exp, y1, y2 in zip(dM.keys(),[8,3,8,2.75,5.5,5.5,5.5,11],[2,2.5,1.5,1.5,2,4,3,4.5]):
    
    #Zonal plots
    ax1 = plt.subplot(8,2,n+1)

    plt.plot(np.sin(fbct.lat.values*np.pi/180),dZ[exp],c=dlc[exp],lw=2,zorder=11)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-1*dZe[exp],1*dZe[exp],facecolor='.75',label=r'$1\sigma$',zorder=-1)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-2*dZe[exp],2*dZe[exp],facecolor='.85',label=r'$2\sigma$',zorder=-2)
    plt.fill_between(np.sin(fbct.lat.values*np.pi/180),-3*dZe[exp],3*dZe[exp],facecolor='.95',label=r'$3\sigma$',zorder=-3)
    
    plt.plot(np.sin(fbct.lat.values*np.pi/180),3*dZe[exp],'k',lw=1,ls='dotted')
    plt.plot(np.sin(fbct.lat.values*np.pi/180),-3*dZe[exp],'k',lw=1,ls='dotted')

    plt.plot([-2,2],[0,0],'k--',lw=1,zorder=10)

    plt.xlim(-1,1)
    plt.ylim(-y1,y1)
    
    plt.xticks(np.sin(np.pi/180*np.arange(-90,91,15)),[r'90$\degree$S',r'',r'',r'',r'30$\degree$S',r'',r'0$\degree$',r'',r'30$\degree$N',r'',r'',r'',r'90$\degree$N'],fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.ylabel(ylab[int(n/2)],fontsize=fs)
    
    if n == 0: 
        plt.title('Zonal trend or difference (%s)\n' % r'$\mathrm{W/m^2}$',fontsize=fs)
        plt.legend(fontsize=fs-8,ncol=3,loc=8,frameon=False)
    
    ax1.text(.5,.825,s=text[int(n/2)],transform = ax1.transAxes,fontsize=fs-6,ha='center')
    
    ax1.text(-.5,.45,s=exp,transform = ax1.transAxes,fontsize=fs,fontweight='bold')
    
    ax1.text(-.225,1,s='(%s)' % lab[n],transform = ax1.transAxes,fontsize=fs-2,fontweight='bold')

    #Bar plots
    ax2 = plt.subplot(8,2,n+2)

    bdata = dM[exp]['NH-SH']
    yerr = 2*dE[exp]['NH-SH']

    bars = plt.bar(np.arange(len(bdata)),bdata,yerr=yerr,width=.67,color=bcolors,edgecolor=bedges,ecolor=ecolors,lw=2,capsize=0)

    for bar, hatch in zip(bars.patches, ['','.','/']+10*['']):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)

    plt.scatter(np.arange(len(bdata))-.18,dM[exp]['NH'],s=25,facecolor='k',edgecolors='w',lw=.5,marker='^',label='NH')
    plt.scatter(np.arange(len(bdata))-.18,dM[exp]['SH'],s=25,facecolor='k',edgecolors='w',lw=.5,marker='v',label='SH')
    plt.scatter(np.arange(len(bdata))+.18,(dM[exp]['NHtr']-dM[exp]['SHtr'])/2,s=25,facecolor='k',edgecolors='w',lw=.5,marker='P',label='Tr')
    plt.scatter(np.arange(len(bdata))+.18,(dM[exp]['NHex']-dM[exp]['SHex'])/2,s=25,facecolor='k',edgecolors='w',lw=.5,marker='X',label='Ex')

    plt.plot([-7,7],[0,0],'k--',lw=1,zorder=0)

    plt.xlim(-.67,5.67)
    plt.ylim(-y2,y2)
    
    plt.xticks(np.arange(len(bdata)),labels,fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    
    if n == 0: 
        plt.title('Hemispheric asymmetry (%s)\n' % r'$\mathrm{W/m^2}$',fontsize=fs)
        plt.legend(fontsize=fs-8,ncol=4,loc=4)
    
    ax2.text(-.225,1,s='(%s)' % lab[n+1],transform = ax2.transAxes,fontsize=fs-2,fontweight='bold')
    
    n += 2

plt.tight_layout()

plt.savefig(dir_fig+'Fig3.png',dpi=450)











