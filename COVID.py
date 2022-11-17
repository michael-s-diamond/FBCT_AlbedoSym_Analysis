"""
Script to analyze what 2020 can teach us about the maintenance of Earth's albedo symmetry
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from glob import glob
import os
import warnings

#Load data
fbct = xr.open_dataset('/Users/mdiamond/Data/CERES/FluxByCldType/FBCT_decomposition.nc')
fTr = xr.open_dataset('/Users/mdiamond/Data/CERES/FluxByCldType/FBCT_climo.nc')
geo = xr.open_dataset('/Users/mdiamond/Data/CERES/FluxByCldType/FBCT_geo_trends.nc')

kTr = xr.open_dataset('/Users/mdiamond/Data/CERES/EBAF/CLR_decomposition_trends_Y.nc')

#Time utilities
months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(len(fbct.time))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)


"""
Analysis of how/whether 2020 looks different from other years
"""
#Prepare arrays
fTr['year'] = np.arange(2003,2022)
ny = len(fTr.year)
nr = len(fTr.reg)
nc = len(fTr.cld)
nt = len(fTr.time)

#
###Calculate detrended values
#
fTr['dR'] = (['reg','time'],np.nan*np.ones((nr,nt))) 
fTr['dR_atm'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt))) 
fTr['dR*_atm'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt))) 
fTr['dR*_clr'] = (['reg','time'],np.nan*np.ones((nr,nt))) 
fTr['dR*_sfc'] = (['reg','time'],np.nan*np.ones((nr,nt))) 
fTr['dR_sfc'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt))) 

fTr['dR'] = fTr['Ra']-(np.array(fTr['Ra_int'])[:,np.newaxis]+np.array(fTr['Ra_trend'])[:,np.newaxis]/120*np.arange(nt)[np.newaxis,:])
fTr['dR_atm'] = fTr['Ra_atm']-(np.array(fTr['Ra_atm_int'])[:,:,np.newaxis]+np.array(fTr['Ra_atm_trend'])[:,:,np.newaxis]/120*np.arange(nt)[np.newaxis,np.newaxis,:])
fTr['dR*_atm'] = fTr['R*a_atm']-(np.array(fTr['R*a_atm_int'])[:,:,np.newaxis]+np.array(fTr['R*a_atm_trend'])[:,:,np.newaxis]/120*np.arange(nt)[np.newaxis,np.newaxis,:])
fTr['dR*_clr'] = fTr['R*a_clr']-(np.array(fTr['R*a_clr_int'])[:,np.newaxis]+np.array(fTr['R*a_clr_trend'])[:,np.newaxis]/120*np.arange(nt)[np.newaxis,:])
fTr['dR*_sfc'] = fTr['R*a_sfc']-(np.array(fTr['R*a_sfc_int'])[:,np.newaxis]+np.array(fTr['R*a_sfc_trend'])[:,np.newaxis]/120*np.arange(nt)[np.newaxis,:])
fTr['dR_sfc'] = fTr['Ra_sfc']-(np.array(fTr['Ra_sfc_int'])[:,np.newaxis]+np.array(fTr['Ra_sfc_trend'])[:,np.newaxis]/120*np.arange(nt)[np.newaxis,:])

#
###Calculate annual averages
#
fTr['Ry'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['Ry_atm'] = (['cld','reg','year'],np.nan*np.ones((nc,nr,ny))) 
fTr['R*y_atm'] = (['cld','reg','year'],np.nan*np.ones((nc,nr,ny))) 
fTr['R*y_clr'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['R*y_sfc'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['Ry_sfc'] = (['cld','reg','year'],np.nan*np.ones((nc,nr,ny))) 

fTr['dRy'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['dRy_atm'] = (['cld','reg','year'],np.nan*np.ones((nc,nr,ny))) 
fTr['dR*y_atm'] = (['cld','reg','year'],np.nan*np.ones((nc,nr,ny))) 
fTr['dR*y_clr'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['dR*y_sfc'] = (['reg','year'],np.nan*np.ones((nr,ny))) 
fTr['dRy_sfc'] = (['reg','year'],np.nan*np.ones((nr,ny))) 

#Fill in values
for iy in range(len(fTr.year)):
    year = fTr.year[iy]
    tmask = fTr.time.dt.year == year
    Mwts_ = Mwts[tmask]
    
    fTr['Ry'][:,iy] = np.average(fTr['R'][:,tmask],weights=Mwts_,axis=-1)
    fTr['Ry_atm'][:,:,iy] = np.average(fTr['R_atm'][:,:,tmask],weights=Mwts_,axis=-1)
    fTr['R*y_atm'][:,:,iy] = np.average(fTr['R*_atm'][:,:,tmask],weights=Mwts_,axis=-1)
    fTr['R*y_clr'][:,iy] = np.average(fTr['R*_clr'][:,tmask],weights=Mwts_,axis=-1)
    fTr['R*y_sfc'][:,iy] = np.average(fTr['R*_sfc'][:,tmask],weights=Mwts_,axis=-1)
    fTr['Ry_sfc'][:,:,iy] = np.average(fTr['R_sfc'][:,:,tmask],weights=Mwts_,axis=-1)
    
    fTr['dRy'][:,iy] = np.average(fTr['dR'][:,tmask],weights=Mwts_,axis=-1)
    fTr['dRy_atm'][:,:,iy] = np.average(fTr['dR_atm'][:,:,tmask],weights=Mwts_,axis=-1)
    fTr['dR*y_atm'][:,:,iy] = np.average(fTr['dR*_atm'][:,:,tmask],weights=Mwts_,axis=-1)
    fTr['dR*y_clr'][:,iy] = np.average(fTr['dR*_clr'][:,tmask],weights=Mwts_,axis=-1)
    fTr['dR*y_sfc'][:,iy] = np.average(fTr['dR*_sfc'][:,tmask],weights=Mwts_,axis=-1)
    fTr['dRy_sfc'][:,iy] = np.average(fTr['dR_sfc'][:,tmask],weights=Mwts_,axis=-1)
    


#Big issue: CERES EBAF and FBCT differ dramatically in terms of clear-sky atm/sfc breakdown in 2020
#Has to come from transmissivity I think?? So the difference is SYN versus EBAF


#
###Plot of detrended annual anomalies for 2020
#

plt.figure('2020anoms',)
plt.clf()
fs = 16

data = [fTr['dRy'].sel(reg='NH-SH',year=2020).values,fTr['dR*y_clr'].sel(reg='NH-SH',year=2020).values,fTr['dR*y_sfc'].sel(reg='NH-SH',year=2020).values,fTr['dR*y_atm'].sel(cld='clr',reg='NH-SH',year=2020).values,fTr['dRy_sfc'].sel(reg='NH-SH',year=2020).values]+[fTr['dRy_atm'].sel(cld=cld,reg='NH-SH',year=2020).values for cld in fTr.cld]

xlab = ['R','*clr','*sfc','*aer','sfc','aer','Cu','Sc','St','Ac','As','Ns','Ci','Cs','Cb']

plt.scatter(np.arange(len(data)),data)
plt.xticks(np.arange(len(data)),xlab)




plt.figure('2020anomsNH')
plt.clf()
fs = 16

dataNH = [fTr['dRy'].sel(reg='NH',year=2020).values,fTr['dR*y_clr'].sel(reg='NH',year=2020).values,fTr['dR*y_sfc'].sel(reg='NH',year=2020).values,fTr['dR*y_atm'].sel(cld='clr',reg='NH',year=2020).values,fTr['dRy_sfc'].sel(reg='NH',year=2020).values]+[fTr['dRy_atm'].sel(cld=cld,reg='NH',year=2020).values for cld in fTr.cld]

plt.scatter(np.arange(len(data)),dataNH)

dataSH = [fTr['dRy'].sel(reg='SH',year=2020).values,fTr['dR*y_clr'].sel(reg='SH',year=2020).values,fTr['dR*y_sfc'].sel(reg='SH',year=2020).values,fTr['dR*y_atm'].sel(cld='clr',reg='SH',year=2020).values,fTr['dRy_sfc'].sel(reg='SH',year=2020).values]+[fTr['dRy_atm'].sel(cld=cld,reg='SH',year=2020).values for cld in fTr.cld]

plt.scatter(np.arange(len(data)),dataSH)
plt.xticks(np.arange(len(data)),xlab)



#Also look at just March-August period??

#Need to look at global maps of differences









