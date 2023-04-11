"""
Make files for each cloud type

Modification history
--------------------
14 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
"""

#Import libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
import os
import warnings

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'

#Load data
cFlx = xr.open_mfdataset(sorted(glob(dir_data+'CERES/FluxByCldType/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_*nc')))


"""
Define cloud types
"""
#
###By thickness
#
thin = cFlx.opt <= 1 #0.02-3.55
midt = np.logical_and(cFlx.opt>1,cFlx.opt<4) #3.55-22.63
thick = cFlx.opt >= 4 #22.63-378.65

dOpt = {'Cu' : thin,  'Ac' : thin,  'Ci' : thin,
        'Sc' : midt,  'As' : midt,  'Cs' : midt,
        'St' : thick, 'Ns' : thick, 'Cb' : thick}

#
###By cloud top height
#
low = cFlx.press <=1 #1000-680 hPa
midp = np.logical_and(cFlx.press > 1, cFlx.press < 4) #680-440 hPa
high = cFlx.press >= 4 #440-10 hPa

dPress = {'Cu' : low, 'Ac' : midp, 'Ci' : high,
          'Sc' : low, 'As' : midp, 'Cs' : high,
          'St' : low, 'Ns' : midp, 'Cb' : high}

"""
Create files for each cloud type separately
"""
for cld in dOpt.keys():
    print('Starting',cld,'...')
    
    #Set up array for dataset
    ds = xr.Dataset()
    ds['time'] = cFlx.time
    ds['lat'] = cFlx.lat
    ds['lon'] = cFlx.lon

    #Calculate relative frequency of occurrence within broader cloud type
    cf = cFlx['cldarea_cldtyp_mon'][dOpt[cld],dPress[cld]]
    rfoc = (cf/(cf.sum(axis=(0,1))+1e-20)).values
    
    #Fill in data
    for var in cFlx:
        if 'cldtyp' in var:
            print(var)
            with warnings.catch_warnings(): #Ignore runtime warnings
                warnings.simplefilter("ignore")
                if 'cldarea' in var:
                    #Sum over cloud fraction bins
                    cld_avg = (cFlx[var][dOpt[cld],dPress[cld]]).sum(axis=(0,1)).values
                    ds[var] = (['time','lat','lon'],cld_avg)
                    ds[var].attrs = cFlx[var].attrs
                else:
                    #Take average weighted by cloud fraction in each bin
                    cld_avg = (cFlx[var][dOpt[cld],dPress[cld]]*rfoc).sum(axis=(0,1)).values
                    ds[var] = (['time','lat','lon'],cld_avg)
                    ds[var].attrs = cFlx[var].attrs
        
    #Save file
    filename = dir_data+'CERES/FluxByCldType/FBCT_cldtype_%s_mon.nc' % cld
    os.system('rm %s' % filename) #Delete file if it already exists
    ds.to_netcdf(path=filename,mode='w')
    print('Finished',cld,'!\n')