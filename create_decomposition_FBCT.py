"""
Make file with breakdowns of atmospheric and surface reflection using the decomposiiton method of Donohoe & Battisti (2011), J. Clim.

Donohoe, A., & Battisti, D. S. (2011). Atmospheric and Surface Contributions to Planetary Albedo. Journal of Climate, 24(16), 4402-4418. 

Modification history
--------------------
14 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
06 May 2024: Michael Diamond, Tallahassee, FL
    -Eliminating regression for transmissivity and using CRE method instead
    -Separate SYN and EBAF sfc/aer decompositions for clear-sky fluxes
13 May 2024: Michael Diamond, flying to Norfolk, VA
    -Fixing system versus component transmissivity issue
17 June 2024: Michael Diamond, Tallahassee, FL
    -Simplifying clear-sky breakdown calculations
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
import scipy.special as sp
import matplotlib.pyplot as plt
from glob import glob
import os
import warnings

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'

#Load data
cFlx = xr.open_mfdataset(sorted(glob(dir_data+'CERES/FluxByCldType/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_*nc'))).sel(time=slice(np.datetime64('2002-07-01'),np.datetime64('2022-06-30')))

cldtypes = ['Cu','Sc','St','Ac','As','Ns','Ci','Cs','Cb']
dFlx = {} #Each cloud type aggregated separately
for cldtype in cldtypes:
    dFlx[cldtype] = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_cldtype_%s_mon.nc' % cldtype).sel(time=slice(np.datetime64('2002-07-01'),np.datetime64('2022-06-30')))

ebaf = xr.open_mfdataset(sorted(glob(dir_data+'CERES/EBAFed42/CERES_EBAF_Ed4.2_Subset_200003-202312.nc'))).sel(time=slice(np.datetime64('2002-07-01'),np.datetime64('2022-06-30')))


"""
Save files with each cloud type broken down by atmosphere and surface components
"""

#Set up master array to store decompositions
xR = xr.Dataset() 
xR['time'] = cFlx.time
xR['lat'] = cFlx.lat
xR['lon'] = cFlx.lon
xR['month'] = np.arange(1,13)
xR['month'].attrs = {'units' : '1', 'long_name' : 'Month of the year'}

#
###All-sky total
#

R = cFlx['toa_sw_all_mon']
R = R.where(R>0,0)

xR['R'] = R
xR['R'].attrs = {'units' : 'W m-2', 'long_name' : 'All-sky reflection'}

S = cFlx['toa_solar_all_mon']

xR['S'] = S
xR['S'].attrs = {'units' : 'W m-2', 'long_name' : 'Insolation'}

#
###Clear-sky
#

R_clr = cFlx['toa_sw_clr_mon']
R_clr = R_clr.where(R_clr>0,0)

xR['R_clr'] = R_clr
xR['R_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky reflection'}

#Use EBAF values to help with decomposition

rsdt = ebaf['solar_mon']
rsut = ebaf['toa_sw_clr_t_mon']
rsds = ebaf['sfc_sw_down_clr_t_mon']
rsus = ebaf['sfc_sw_up_clr_t_mon']

R_aer = S*(rsdt*rsut-rsds*rsus)/(rsdt**2-rsus**2)
R_aer = R_aer.where(R_aer>0,0)
R_aer = R_aer.where(R_aer<R_clr,R_clr)

xR['R_aer'] = R_aer
xR['R_aer'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky atmospheric reflection'}

xR['R_sfc'] = xR['R_clr']-xR['R_aer']
xR['R_sfc'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky surface reflection'}

#
###Cloud types
#

for i in range(len(cldtypes)):
    ct = cldtypes[i]

    R_cld = dFlx[ct]['toa_sw_cldtyp_mon']
    C = dFlx[ct]['cldarea_cldtyp_mon']/100

    xR['R_%s' % ct] = C*(R_cld-R_clr) #Positive up
    xR['R_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Cloud radiative effect for %s scenes' % ct}


#
###Calculate climatological values
#

Feb_wts = [28]+4*[29,28,28,28]+[29,28,28] #leap years

print('Calculating:')

#
###All-sky reflection and insolation
#
print('...all...')
xR['Rc'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['Rc'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of R'}
xR['Ra'] = (['time','lat','lon'],np.nan*np.ones(xR['R'].shape))
xR['Ra'].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of R'}

xR['Sc'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['Sc'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of S'}
xR['Sa'] = (['time','lat','lon'],np.nan*np.ones(xR['R'].shape))
xR['Sa'].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of S'}

for m in range(1,13):
   
    R = xR['R'].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
    if m ==2: xR['Rc'][m-1] = np.average(R[R.time.dt.month==m],weights=Feb_wts,axis=0)
    else: xR['Rc'][m-1] = R[R.time.dt.month==m].mean(axis=0)
    xR['Ra'][xR.time.dt.month==m] = xR['R'][xR.time.dt.month==m].values-xR['Rc'][m-1].values
    
    SS = xR['S'].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
    if m ==2: xR['Sc'][m-1] = np.average(SS[SS.time.dt.month==m],weights=Feb_wts,axis=0)
    else: xR['Sc'][m-1] = SS[SS.time.dt.month==m].mean(axis=0)
    xR['Sa'][xR.time.dt.month==m] = xR['S'][xR.time.dt.month==m].values-xR['Sc'][m-1].values


print('...clr...')

#Climatology for S*Aaer and S*Asfc
for loc in ['clr','aer','sfc']:
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        xR['Rc_%s' % loc] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
        xR['Rc_%s' % loc].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of R%s' % loc}
        xR['Ra_%s' % loc] = (['time','lat','lon'],np.nan*np.ones(xR['R'].shape))
        xR['Ra_%s' % loc].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of R%s' % loc}

        for m in range(1,13):

            R = xR['R_%s' % (loc)].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
            if m ==2: xR['Rc_%s' % (loc)][m-1] = np.average(R[R.time.dt.month==m],weights=Feb_wts,axis=0)
            else: xR['Rc_%s' % (loc)][m-1] = R[R.time.dt.month==m].mean(axis=0)
            xR['Ra_%s' % (loc)][xR.time.dt.month==m] = xR['R_%s' % (loc)][xR.time.dt.month==m].values-xR['Rc_%s' % (loc)][m-1].values

#Climatology for R by cloud scene
for ct in cldtypes:
    print('...'+ct+'...')

    xR['Rc_%s' % ct] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
    xR['Rc_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of R for %s scenes' % ct}
    xR['Ra_%s' % ct] = (['time','lat','lon'],np.nan*np.ones(xR['R'].shape))
    xR['Ra_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of reflection for %s scenes' % ct}

    for m in range(1,13):

        Ratm = xR['R_%s' % ct].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
        if m ==2: xR['Rc_%s' % ct][m-1] = np.average(Ratm[Ratm.time.dt.month==m],weights=Feb_wts,axis=0)
        else: xR['Rc_%s' % ct][m-1] = Ratm[Ratm.time.dt.month==m].mean(axis=0)
        xR['Ra_%s' % ct][xR.time.dt.month==m] = xR['R_%s' % ct][xR.time.dt.month==m].values-xR['Rc_%s' % ct][m-1].values

print('Done!')
print()
    
#
###Save smaller files to simplify storage/transfer
#

print('Saving files:')

#All-sky values
print('...all...')
tmp = xr.Dataset()
tmp['time'] = cFlx.time
tmp['lat'] = cFlx.lat
tmp['lon'] = cFlx.lon
tmp['month'] = np.arange(1,13)
tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

for var in ['R','Rc','Ra','S','Sc','Sa']:
    tmp[var] = xR[var]

filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_all.nc'
os.system('rm %s' % filename)
tmp.to_netcdf(path=filename,mode='w')
    
#Clear-sky values
print('...clr...')
tmp = xr.Dataset()
tmp['time'] = cFlx.time
tmp['lat'] = cFlx.lat
tmp['lon'] = cFlx.lon
tmp['month'] = np.arange(1,13)
tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

for var in ['R_clr','Rc_clr','Ra_clr',
            'R_aer','Rc_aer','Ra_aer',
            'R_sfc','Rc_sfc','Ra_sfc']:
    tmp[var] = xR[var]

filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_clr.nc'
os.system('rm %s' % filename)
tmp.to_netcdf(path=filename,mode='w')
    
#Cloud-type values
for cld in cldtypes:
    print('...'+cld+'...')
    
    tmp = xr.Dataset()
    tmp['time'] = cFlx.time
    tmp['lat'] = cFlx.lat
    tmp['lon'] = cFlx.lon
    tmp['month'] = np.arange(1,13)
    tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

    for var in ['R_','Rc_','Ra_']:
        tmp[var+cld] = xR[var+cld]

    filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_%s.nc' % cld
    os.system('rm %s' % filename)
    tmp.to_netcdf(path=filename,mode='w')
    
    
print('Done!')
