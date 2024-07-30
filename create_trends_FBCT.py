"""
Create file with globally/regionally averaged EBAF reflection broken down by cloud type and atmospheric and surface components

Modification history
--------------------
14 June 2022: Michael Diamond, Boulder, CO
    -Created
16 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted to new averaging period, simplified regions
07 May 2024: Michael Diamond, Tallahassee, FL
    -Simplified cloud radiative effect definitions
13 May 2024: Michael Diamond, flying to Norfolk, VA
    -Fixing system versus component transmissivity issue
20 June 2024: Michael Diamond, Tallahassee, FL
    -Updated with simpler clear-sky decomposition method
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
from glob import glob
import os
import warnings

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'

#Load data
cFlx = xr.open_mfdataset(sorted(glob(dir_data+'CERES/FluxByCldType/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_*nc'))).sel(time=slice(np.datetime64('2002-07-01'),np.datetime64('2022-06-30'))) #raw
fbct = xr.open_mfdataset(glob(dir_data+'CERES/FluxByCldType/FBCT_decomposition*.nc')) #processed

#Create needed space and time utilities

z_weights = np.genfromtxt(dir_data+'CERES/zone_weights_lou.txt',skip_header=17,skip_footer=1)[:,1]

def geoavg(data,hemi='both',reg='all',W=None):
    """
    Take an area-weighted average over a defined region
    
    Assumes arrays have a time axis and the standard CERES lat, lon dimensions (180 x 360)
    
    Uses CERES zonal weighting accounting for oblate spheroid geometry
    
    Parameters
    ----------
    data : array
    3D (nt, nlat, nlon) array to be averaged
    
    hemi : str
    Either 'NH' or 'SH' to pick a hemisphere, or 'both' for global average
    
    reg : str
    Either 'tr' for the tropics (0-30 deg), 'ex' for the extratropics (30-90 deg), or 'all' for full hemisphere or global average
    
    W : array
    2D (nlat, nlon) array to provide additional weighting factor
    """
    
    lon, lat = np.meshgrid(fbct.lon,fbct.lat)
    zlat = z_weights[np.newaxis,:,np.newaxis]*np.ones(data.shape) #Oblate spheroid weights
    
    mask = lat[:,0]<1e10
    
    #Hemisphere mask
    if hemi == 'NH':
        mask = lat[:,0]>0
    elif hemi == 'SH':
        mask = lat[:,0]<0
    
    #Region mask
    if reg == 'tr':
        mask = np.logical_and(mask,np.abs(lat[:,0])<30)
    elif reg == 'ex':
        mask = np.logical_and(mask,np.abs(lat[:,0])>30)
    
    #Take and return weighted average
    if np.shape(W) != ():
        return np.average(data[:,mask],weights=zlat[:,mask]*W[:,mask],axis=(1,2))
    else:
        return np.average(data[:,mask],weights=zlat[:,mask],axis=(1,2))

months = np.array([np.datetime64('2002-07')+np.timedelta64(i,'M') for i in range(12*(2022-2002))])
Mwts = np.array((months+np.timedelta64(1,'M')+np.timedelta64(1,'D'))-(months+np.timedelta64(1,'D')),dtype=float)

"""
Create file with spatially averaged cloudy- and clear-sky reflection breakdown into surface and atmospheric components
"""

xTr = xr.Dataset()

xTr['time'] = fbct.time
nt = len(xTr.time)

xTr['t'] = np.arange(len(fbct.time))
xTr['t'].attrs = {'long_name' : 'Months since 2002-07-15'}

xTr['month'] = np.arange(1,13)
nm = len(xTr.month)
xTr['month'].attrs = {'long_name' : 'Month for climatological (2002-07 to 2022-06) average'}

cldtypes = ['Cu','Sc','St','Ac','As','Ns','Ci','Cs','Cb']
xTr['cld'] = cldtypes
nc = len(xTr.cld)
xTr['cld'].attrs = {'long_name' : 'Cloud type (ISCCP)'}

xTr['reg'] = ['global','NH','SH','NH-SH','NHex','NHtr','SHtr','SHex']
nr = len(xTr.reg)
xTr['reg'].attrs = {'long_name' : 'NH = Northern Hemisphere, SH = Southern Hemisphere, tr = tropics (0-30 deg), ex = extratropics (30-90 deg)'}


#Reflected SW from atmosphere

xTr['R_cld'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['R_cld'].attrs = {'long_name' : 'Reflected atmospheric SW time series','units' : 'W m-2'}

xTr['Rc_cld'] = (['cld','reg','month'],np.nan*np.ones((nc,nr,nm)))
xTr['Rc_cld'].attrs = {'long_name' : 'Reflected atmospheric SW climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Ra_cld'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['Ra_cld'].attrs = {'long_name' : 'Deseasonalized reflected atmospheric SW anomalies','units' : 'W m-2'}

xTr['Ra_cld_int'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ra_cld_int'].attrs = {'long_name' : 'Rcld anomaly intercept','units' : 'W m-2'}

xTr['Ra_cld_trend'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ra_cld_trend'].attrs = {'long_name' : 'Rcld anomaly trend','units' : 'W m-2 decade-1'}

xTr['Ra_cld_e'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['Ra_cld_e'].attrs = {'long_name' : 'Rcld anomaly residuals','units' : 'W m-2'}

xTr['Ra_cld_nu'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ra_cld_nu'].attrs = {'long_name' : 'Degrees of freedom for the Rcld anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ra_cld_int_err'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ra_cld_int_err'].attrs = {'long_name' : 'Rcld anomaly intercept standard error','units' : 'W m-2'}

xTr['Ra_cld_trend_err'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ra_cld_trend_err'].attrs = {'long_name' : 'Rcld anomaly trend standard error','units' : 'W m-2 decade-1'}

"""#Cloud fraction

xTr['C'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['C'].attrs = {'long_name' : 'Cloud fraction time series','units' : '1'}

xTr['Cc'] = (['cld','reg','month'],np.nan*np.ones((nc,nr,nm)))
xTr['Cc'].attrs = {'long_name' : 'Cloud fraction climatology (2002-07 to 2022-06)','units' : '1'}

xTr['Ca'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['Ca'].attrs = {'long_name' : 'Deseasonalized cloud fraction anomalies','units' : '1'}

xTr['Ca_int'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ca_int'].attrs = {'long_name' : 'C anomaly intercept','units' : '1'}

xTr['Ca_trend'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ca_trend'].attrs = {'long_name' : 'C anomaly trend','units' : '1'}

xTr['Ca_e'] = (['cld','reg','time'],np.nan*np.ones((nc,nr,nt)))
xTr['Ca_e'].attrs = {'long_name' : 'C anomaly residuals','units' : '1'}

xTr['Ca_nu'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ca_nu'].attrs = {'long_name' : 'Degrees of freedom for the C anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ca_int_err'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ca_int_err'].attrs = {'long_name' : 'C anomaly intercept standard error','units' : '1'}

xTr['Ca_trend_err'] = (['cld','reg'],np.nan*np.ones((nc,nr)))
xTr['Ca_trend_err'].attrs = {'long_name' : 'C anomaly trend standard error','units' : '1'}
"""
#Reflected clear-sky SW 

xTr['R_clr'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['R_clr'].attrs = {'long_name' : 'Reflected clear-sky SW time series','units' : 'W m-2'}

xTr['Rc_clr'] = (['reg','month'],np.nan*np.ones((nr,nm)))
xTr['Rc_clr'].attrs = {'long_name' : 'Reflected clear-sky SW climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Ra_clr'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_clr'].attrs = {'long_name' : 'Deseasonalized reflected clear-sky SW anomalies','units' : 'W m-2'}

xTr['Ra_clr_int'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_clr_int'].attrs = {'long_name' : 'Rclr anomaly intercept','units' : 'W m-2'}

xTr['Ra_clr_trend'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_clr_trend'].attrs = {'long_name' : 'Rclr anomaly trend','units' : 'W m-2 decade-1'}

xTr['Ra_clr_e'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_clr_e'].attrs = {'long_name' : 'Rclr anomaly residuals','units' : 'W m-2'}

xTr['Ra_clr_nu'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_clr_nu'].attrs = {'long_name' : 'Degrees of freedom for the Rclr anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ra_clr_int_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_clr_int_err'].attrs = {'long_name' : 'Rclr anomaly intercept standard error','units' : 'W m-2'}

xTr['Ra_clr_trend_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_clr_trend_err'].attrs = {'long_name' : 'Rclr anomaly trend standard error','units' : 'W m-2 decade-1'}

#Reflected SW from the surface

xTr['R_sfc'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['R_sfc'].attrs = {'long_name' : 'Reflected surface SW time series','units' : 'W m-2'}

xTr['Rc_sfc'] = (['reg','month'],np.nan*np.ones((nr,nm)))
xTr['Rc_sfc'].attrs = {'long_name' : 'Reflected surface SW climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Ra_sfc'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_sfc'].attrs = {'long_name' : 'Deseasonalized reflected surface SW anomalies','units' : 'W m-2'}

xTr['Ra_sfc_int'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_sfc_int'].attrs = {'long_name' : 'Rsfc anomaly intercept','units' : 'W m-2'}

xTr['Ra_sfc_trend'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_sfc_trend'].attrs = {'long_name' : 'Rsfc anomaly trend','units' : 'W m-2 decade-1'}

xTr['Ra_sfc_e'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_sfc_e'].attrs = {'long_name' : 'Rsfc anomaly residuals','units' : 'W m-2'}

xTr['Ra_sfc_nu'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_sfc_nu'].attrs = {'long_name' : 'Degrees of freedom for the Rsfc anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ra_sfc_int_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_sfc_int_err'].attrs = {'long_name' : 'Rsfc anomaly intercept standard error','units' : 'W m-2'}

xTr['Ra_sfc_trend_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_sfc_trend_err'].attrs = {'long_name' : 'Rsfc anomaly trend standard error','units' : 'W m-2 decade-1'}

#Reflected SW from the clear-sky atmosphere

xTr['R_aer'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['R_aer'].attrs = {'long_name' : 'Reflected clear-sky atmosphere SW time series','units' : 'W m-2'}

xTr['Rc_aer'] = (['reg','month'],np.nan*np.ones((nr,nm)))
xTr['Rc_aer'].attrs = {'long_name' : 'Reflected clear-sky atmosphere SW climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Ra_aer'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_aer'].attrs = {'long_name' : 'Deseasonalized reflected clear-sky atmosphere SW anomalies','units' : 'W m-2'}

xTr['Ra_aer_int'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_aer_int'].attrs = {'long_name' : 'Raer anomaly intercept','units' : 'W m-2'}

xTr['Ra_aer_trend'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_aer_trend'].attrs = {'long_name' : 'Raer anomaly trend','units' : 'W m-2 decade-1'}

xTr['Ra_aer_e'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_aer_e'].attrs = {'long_name' : 'Raer anomaly residuals','units' : 'W m-2'}

xTr['Ra_aer_nu'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_aer_nu'].attrs = {'long_name' : 'Degrees of freedom for the Raer anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ra_aer_int_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_aer_int_err'].attrs = {'long_name' : 'Raer anomaly intercept standard error','units' : 'W m-2'}

xTr['Ra_aer_trend_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_aer_trend_err'].attrs = {'long_name' : 'Raer anomaly trend standard error','units' : 'W m-2 decade-1'}



#Reflected SW at TOA

xTr['R'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['R'].attrs = {'long_name' : 'Reflected SW time series','units' : 'W m-2'}

xTr['Rc'] = (['reg','month'],np.nan*np.ones((nr,nm)))
xTr['Rc'].attrs = {'long_name' : 'Reflected SW climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Ra'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra'].attrs = {'long_name' : 'Deseasonalized reflected SW anomalies','units' : 'W m-2'}

xTr['Ra_int'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_int'].attrs = {'long_name' : 'R anomaly intercept','units' : 'W m-2'}

xTr['Ra_trend'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_trend'].attrs = {'long_name' : 'R anomaly trend','units' : 'W m-2 decade-1'}

xTr['Ra_e'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Ra_e'].attrs = {'long_name' : 'R anomaly residuals','units' : 'W m-2'}

xTr['Ra_nu'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_nu'].attrs = {'long_name' : 'Degrees of freedom for the R anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Ra_int_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_int_err'].attrs = {'long_name' : 'R anomaly intercept standard error','units' : 'W m-2'}

xTr['Ra_trend_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Ra_trend_err'].attrs = {'long_name' : 'R anomaly trend standard error','units' : 'W m-2 decade-1'}

#Incoming SW at TOA

xTr['S'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['S'].attrs = {'long_name' : 'Insolation time series','units' : 'W m-2'}

xTr['Sc'] = (['reg','month'],np.nan*np.ones((nr,nm)))
xTr['Sc'].attrs = {'long_name' : 'Insolation climatology (2002-07 to 2022-06)','units' : 'W m-2'}

xTr['Sa'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Sa'].attrs = {'long_name' : 'Deseasonalized insolation anomalies','units' : 'W m-2'}

xTr['Sa_int'] = (['reg'],np.nan*np.ones((nr)))
xTr['Sa_int'].attrs = {'long_name' : 'S anomaly intercept','units' : 'W m-2'}

xTr['Sa_trend'] = (['reg'],np.nan*np.ones((nr)))
xTr['Sa_trend'].attrs = {'long_name' : 'S anomaly trend','units' : 'W m-2 decade-1'}

xTr['Sa_e'] = (['reg','time'],np.nan*np.ones((nr,nt)))
xTr['Sa_e'].attrs = {'long_name' : 'S anomaly residuals','units' : 'W m-2'}

xTr['Sa_nu'] = (['reg'],np.nan*np.ones((nr)))
xTr['Sa_nu'].attrs = {'long_name' : 'Degrees of freedom for the S anomaly residuals assuming autoregressive process (Bretherton et al., 1999)','units' : '1'}

xTr['Sa_int_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Sa_int_err'].attrs = {'long_name' : 'S anomaly intercept standard error','units' : 'W m-2'}

xTr['Sa_trend_err'] = (['reg'],np.nan*np.ones((nr)))
xTr['Sa_trend_err'].attrs = {'long_name' : 'S anomaly trend standard error','units' : 'W m-2 decade-1'}

#
###Fill in data
#
print('Starting calculations:')

#Cloud types
for ci in range(len(xTr.cld)):
    cld = xTr.cld[ci].values
    print('...'+str(cld)+'...')
    
    #R and Ra by cloud type
    for var in ['R_cld','Ra_cld']:
        
        fdata = fbct[var.split('_')[0]+'_%s' % cld]

        xTr[var][ci,0,:] = geoavg(fdata)
        xTr[var][ci,1,:] = geoavg(fdata,hemi='NH')
        xTr[var][ci,2,:] = geoavg(fdata,hemi='SH')
        xTr[var][ci,3,:] = geoavg(fdata,hemi='NH')-geoavg(fdata,hemi='SH')
        xTr[var][ci,4,:] = geoavg(fdata,hemi='NH',reg='ex')
        xTr[var][ci,5,:] = geoavg(fdata,hemi='NH',reg='tr')    
        xTr[var][ci,6,:] = geoavg(fdata,hemi='SH',reg='tr')
        xTr[var][ci,7,:] = geoavg(fdata,hemi='SH',reg='ex')
            
    #Rc
    for var in ['Rc_cld']:
        for m in range(1,13):
            fdata = fbct[var.split('_')[0]+'_%s' % cld].sel(month=m).values[np.newaxis,:,:]

            xTr[var][ci,0,m-1] = float(geoavg(fdata))
            xTr[var][ci,1,m-1] = float(geoavg(fdata,hemi='NH'))
            xTr[var][ci,2,m-1] = float(geoavg(fdata,hemi='SH'))
            xTr[var][ci,3,m-1] = float(geoavg(fdata,hemi='NH')-geoavg(fdata,hemi='SH'))
            xTr[var][ci,4,m-1] = float(geoavg(fdata,hemi='NH',reg='ex'))
            xTr[var][ci,5,m-1] = float(geoavg(fdata,hemi='NH',reg='tr'))
            xTr[var][ci,6,m-1] = float(geoavg(fdata,hemi='SH',reg='tr'))
            xTr[var][ci,7,m-1] = float(geoavg(fdata,hemi='SH',reg='ex'))
    
#All others 
print('...else...')
for var in ['R','Ra','S','Sa','R_clr','Ra_clr','R_aer','Ra_aer','R_sfc','Ra_sfc']:
    
    fdata = fbct[var]
            
    xTr[var][0,:] = geoavg(fdata)
    xTr[var][1,:] = geoavg(fdata,hemi='NH')
    xTr[var][2,:] = geoavg(fdata,hemi='SH')
    xTr[var][3,:] = geoavg(fdata,hemi='NH')-geoavg(fdata,hemi='SH')
    xTr[var][4,:] = geoavg(fdata,hemi='NH',reg='ex')
    xTr[var][5,:] = geoavg(fdata,hemi='NH',reg='tr')
    xTr[var][6,:] = geoavg(fdata,hemi='SH',reg='tr')
    xTr[var][7,:] = geoavg(fdata,hemi='SH',reg='ex')

for var in ['Rc','Sc','Rc_clr','Rc_aer','Rc_sfc']:
    for m in range(1,13):
    
        fdata = fbct[var].sel(month=m).values[np.newaxis,:,:]
        
        xTr[var][0,m-1] = float(geoavg(fdata))
        xTr[var][1,m-1] = float(geoavg(fdata,hemi='NH'))
        xTr[var][2,m-1] = float(geoavg(fdata,hemi='SH'))
        xTr[var][3,m-1] = float(geoavg(fdata,hemi='NH')-geoavg(fdata,hemi='SH'))
        xTr[var][4,m-1] = float(geoavg(fdata,hemi='NH',reg='ex'))
        xTr[var][5,m-1] = float(geoavg(fdata,hemi='NH',reg='tr'))
        xTr[var][6,m-1] = float(geoavg(fdata,hemi='SH',reg='tr'))
        xTr[var][7,m-1] = float(geoavg(fdata,hemi='SH',reg='ex'))


#
###Regressions for cloud types
#

for var in ['Ra_cld']:

    #Set independent and dependent variables and means
    y = xTr[var]
    ybar = y.mean(axis=-1)
    x = xTr['t'].values
    xbar = x.mean()

    #Calculate slope and intercept
    beta_hat = np.sum((x-xbar)*(y-ybar),axis=-1)/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar

    xTr[var+'_trend'][:,:] = 12*10*beta_hat.values
    xTr[var+'_int'][:,:] = alpha_hat.values

    #Calculate residuals and DOF of residuals
    yhat = alpha_hat.values[:,:,np.newaxis]+beta_hat.values[:,:,np.newaxis]*x[np.newaxis,np.newaxis,:]
    e = y-yhat

    xx = e[:,:,1:].values
    yy = e[:,:,:-1].values
    xm = xx.mean(axis=-1)[:,:,np.newaxis]*np.ones(xx.shape)
    ym = yy.mean(axis=-1)[:,:,np.newaxis]*np.ones(yy.shape)

    r1e_num = np.sum((xx-xm)*(yy-ym),axis=-1)
    r1e_denom = np.sum((xx-xm)**2,axis=-1)*np.sum((yy-ym)**2,axis=-1)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = nt*(1-r1e)/(1+r1e)
    nue[nue>nt] = nt #Can't have DOF larger than nt

    xTr[var+'_e'][:,:,:] = e
    xTr[var+'_nu'][:,:] = nue

    #Calculate slope and intercept uncertainty
    sbh = np.sqrt(np.sum(e**2,axis=-1)/(nue-2)/np.sum((x-xbar)**2))
    sah = sbh*np.sqrt(np.sum(x**2)/nue)

    xTr[var+'_trend_err'][:,:] = 12*10*sbh
    xTr[var+'_int_err'][:,:] = sah

#
###Regressions for other variables
#

for var in ['Ra','Sa','Ra_clr','Ra_sfc','Ra_aer']:

    #Set independent and dependent variables and means
    y = xTr[var]
    ybar = y.mean(axis=-1)
    x = xTr['t'].values
    xbar = x.mean()

    #Calculate slope and intercept
    beta_hat = np.sum((x-xbar)*(y-ybar),axis=-1)/np.sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar

    xTr[var+'_trend'][:] = 12*10*beta_hat.values
    xTr[var+'_int'][:] = alpha_hat.values

    #Calculate residuals and DOF of residuals
    yhat = alpha_hat.values[:,np.newaxis]+beta_hat.values[:,np.newaxis]*x[np.newaxis,:]
    e = y-yhat

    xx = e[:,1:].values
    yy = e[:,:-1].values
    xm = xx.mean(axis=-1)[:,np.newaxis]*np.ones(xx.shape)
    ym = yy.mean(axis=-1)[:,np.newaxis]*np.ones(yy.shape)

    r1e_num = np.sum((xx-xm)*(yy-ym),axis=-1)
    r1e_denom = np.sum((xx-xm)**2,axis=-1)*np.sum((yy-ym)**2,axis=-1)
    r1e = r1e_num/np.sqrt(r1e_denom)
    nue = nt*(1-r1e)/(1+r1e)
    nue[nue>nt] = nt #Can't have DOF larger than nt

    xTr[var+'_e'][:,:] = e
    xTr[var+'_nu'][:] = nue

    #Calculate slope and intercept uncertainty
    sbh = np.sqrt(np.sum(e**2,axis=-1)/(nue-2)/np.sum((x-xbar)**2))
    sah = sbh*np.sqrt(np.sum(x**2)/nue)

    xTr[var+'_trend_err'][:] = 12*10*sbh
    xTr[var+'_int_err'][:] = sah

#
###Save file
#
filename = dir_data+'CERES/FluxByCldType/FBCT_gavg_trends.nc'
os.system('rm %s' % filename)
xTr.to_netcdf(path=filename,mode='w')

print('Done!')
