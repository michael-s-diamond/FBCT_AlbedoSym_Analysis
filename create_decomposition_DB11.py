"""
Make file with breakdowns of atmospheric and surface reflection using the decomposiiton method of Donohoe & Battisti (2011), J. Clim.

Donohoe, A., & Battisti, D. S. (2011). Atmospheric and Surface Contributions to Planetary Albedo. Journal of Climate, 24(16), 4402-4418. 

Modification history
--------------------
14 March 2023: Michael Diamond, Tallahassee, FL
    -Adapted from previous version
"""

#Import libraries
import numpy as np
import xarray as xr
import scipy
from scipy import stats
import scipy.special as sp
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
from glob import glob
import os
import warnings

#Set paths
dir_data = '/Users/michaeldiamond/Documents/Data/'

#Load data
cFlx = xr.open_mfdataset(sorted(glob(dir_data+'CERES/FluxByCldType/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_*nc')))

cldtypes = ['Cu','Sc','St','Ac','As','Ns','Ci','Cs','Cb']
dFlx = {} #Each cloud type aggregated separately
for cldtype in cldtypes:
    dFlx[cldtype] = xr.open_dataset(dir_data+'CERES/FluxByCldType/FBCT_cldtype_%s_mon.nc' % cldtype)

syn = xr.open_mfdataset(sorted(glob(dir_data+'CERES/SYN1deg/CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4.1_Subset_*nc')))
syn['T'] = syn['adj_atmos_sw_down_all_surface_mon']/syn['toa_solar_all_mon']

"""
Regression of cloud fraction (+ SZA and elevation) against system transmissivity
"""

#Get features and target
dX_ = {}
for ct in cldtypes:
    dX_[ct] = np.ravel(dFlx[ct]['cldarea_cldtyp_mon']/100)
dX_['SZA'] = np.ravel(np.cos(cFlx['solar_zen_angle_mon']*np.pi/180))
dX_['z'] = np.ravel(syn['sfc_elev_mon']/1000)
T_ = np.ravel(syn['T'])

#Clean up NaNs
nan = np.zeros(len(dX_['z']))
for key in dX_.keys():
    nan += np.array(np.isnan(dX_[key]),dtype=int)
nan += np.array(np.isnan(T_),dtype=int)
valid = nan == 0

dX = {}
for key in dX_.keys():
    dX[key] = dX_[key][valid]
T = T_[valid]

#Run regression of logit(T) versus features
X = np.array([dX[key] for key in dX.keys()]).T
y = sp.logit(T)

reg = LR().fit(X=X,y=y)

R2 = reg.score(X=X,y=y)

yhat = reg.predict(X)

That = sp.expit(yhat)


"""
Save files with each cloud type broken down by atmosphere and surface components
"""

def a_atm_solver(A,a,T,sign=-1):
    """
    Calculate atmospheric contribution to scene albedo.
    
    Parameters
    ----------
    A : array-like
    Scene albedo
    
    a : array-like
    Surface albedo
    
    T : array-like
    Atmospheric transmissivity
    
    sign : -1 or 1
    """
    b = a**2*(4*T**2+A**2)-2*a*A+1
    return (sign*np.sqrt(b)+a*A+1)/(2*a)

a_sfc = np.array(syn['adj_atmos_sw_up_clr_surface_mon']/syn['adj_atmos_sw_down_clr_surface_mon'])
a_sfc[np.isnan(a_sfc)] = 0

S = np.array(cFlx['toa_solar_all_mon'])

xR = xr.Dataset() #Set up master array to store decompositions
xR['time'] = cFlx.time
xR['lat'] = cFlx.lat
xR['lon'] = cFlx.lon
xR['month'] = np.arange(1,13)
xR['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

#
###Transmissivity
#

xR['T_all'] = (['time','lat','lon'],syn['T'].values)
xR['T_all'].attrs = {'units' : '1', 'long_name' : 'All-sky atmospheric transmissivity'}

y_hat = np.array(reg.intercept_+reg.coef_[-2]*np.cos(cFlx['solar_zen_angle_mon']*np.pi/180)+reg.coef_[-1]*syn['sfc_elev_mon']/1000)
for i in range(len(cldtypes)):
    ct = cldtypes[i]
    y_hat += np.array(reg.coef_[i]*dFlx[ct]['cldarea_cldtyp_mon']/100)

T_hat = sp.expit(y_hat)

xR['T_hat'] = (['time','lat','lon'],T_hat)
xR['T_hat'].attrs = {'units' : '1', 'long_name' : 'Estimated all-sky atmospheric transmissivity'}


#
###Clear-sky
#

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    T_clr = np.array(syn['adj_atmos_sw_down_clr_surface_mon']/syn['toa_solar_all_mon'])
    T_clr[np.isnan(T_clr)] = 0

    A_clr = np.array(cFlx['toa_sw_clr_mon']/cFlx['toa_solar_all_mon'])
    A_clr[np.isnan(A_clr)] = 0

    #Solve a_atm equaiton for a_sfc != 0 and then fill in solution for a_sfc == 0
    a_atm_ = a_atm_solver(A=A_clr,a=a_sfc,T=T_clr,sign=-1)

    a_atm_tofill = np.copy(a_atm_)
    a_atm_tofill[a_sfc==0] = 0
    fill = np.copy(A_clr)
    fill[a_sfc!=0] = 0
    a_atm_clr = a_atm_tofill+fill

    #Fix invalid values
    a_atm_clr[a_atm_clr<0] = 0
    a_atm_clr[a_atm_clr>1] = 1

    #Get total cloud fraction
    C = np.array(cFlx['cldarea_total_mon'])/100
    C[np.isnan(C)] = 0
    
    #Fill in file
    xR['T_clr'] = (['time','lat','lon'],T_clr)
    xR['T_clr'].attrs = {'units' : '1', 'long_name' : 'Clear-sky atmospheric transmissivity'}
    
    xR['S*A_clr'] = (['time','lat','lon'],S*A_clr)
    xR['S*A_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky reflection assuming perpetual cloud-free conditions'}
    
    xR['R_clr'] = (['time','lat','lon'],S*(1-C)*A_clr)
    xR['R_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky reflection'}

    xR['S*A_atm_clr'] = (['time','lat','lon'],S*a_atm_clr)
    xR['S*A_atm_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky atmospheric reflection assuming perpetual cloud-free conditions'}
    
    xR['R_atm_clr'] = (['time','lat','lon'],S*(1-C)*a_atm_clr)
    xR['R_atm_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky atmospheric reflection'}

    xR['S*A_sfc_clr'] = (['time','lat','lon'],S*a_sfc*T_clr**2/(1-a_sfc*a_atm_clr))
    xR['S*A_sfc_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky surface reflection assuming perpetural cloud-free conditions'}
    
    xR['R_sfc_clr'] = (['time','lat','lon'],S*(1-C)*a_sfc*T_clr**2/(1-a_sfc*a_atm_clr))
    xR['R_sfc_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Clear-sky surface reflection'}


#
###Cloud types
#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    for i in range(len(cldtypes)):
        ct = cldtypes[i]
        print(ct)

        T_cld = np.array(sp.expit(reg.intercept_+reg.coef_[i]+reg.coef_[-2]*np.cos(cFlx['solar_zen_angle_mon']*np.pi/180)+reg.coef_[-1]*syn['sfc_elev_mon']/1000))
        T_cld[np.isnan(T_cld)] = 0

        A_cld = np.array(dFlx[ct]['toa_sw_cldtyp_mon']/S)
        A_cld[np.isnan(A_cld)] = 0

        C = np.array(dFlx[ct]['cldarea_cldtyp_mon']/100)
        C[np.isnan(C)] = 0

        #Solve a_atm equaiton for a_sfc != 0 and then fill in solution for a_sfc == 0
        a_atm_ = a_atm_solver(A=A_cld,a=a_sfc,T=T_cld,sign=-1)

        a_atm_tofill = np.copy(a_atm_)
        a_atm_tofill[a_sfc==0] = 0
        fill = np.copy(A_cld)
        fill[a_sfc!=0] = 0
        a_atm_cld = a_atm_tofill+fill

        #Fix invalid values
        a_atm_cld[a_atm_cld<0] = 0
        a_atm_cld[a_atm_cld>1] = 1

        #Fill in file
        xR['T_%s' % ct] = (['time','lat','lon'],T_cld)
        xR['T_%s' % ct].attrs = {'units' : '1', 'long_name' : 'Atmospheric transmissivity for %s scenes' % ct}
        
        xR['C_%s' % ct] = (['time','lat','lon'],C)
        xR['C_%s' % ct].attrs = {'units' : '1', 'long_name' : '%s cloud fraction' % ct}

        xR['R_atm_%s' % ct] = (['time','lat','lon'],S*C*a_atm_cld)
        xR['R_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Atmospheric reflection for %s scenes' % ct}
        
        xR['S*A_atm_%s' % ct] = (['time','lat','lon'],S*a_atm_cld)
        xR['S*A_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Atmospheric reflection assuming perpetual %s scenes' % ct}

        xR['R_sfc_%s' % ct] = (['time','lat','lon'],S*C*a_sfc*T_cld**2/(1-a_sfc*a_atm_cld))
        xR['R_sfc_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Surface reflection for %s scenes' % ct}


#
###Calculate climatological values
#

Feb_wts = [28]+4*[29,28,28,28]+[29,28,28] #leap years

#Climatology for total Rsfc
R_sfc = np.zeros(xR['T_all'].shape)
for ct in ['clr']+cldtypes: R_sfc += xR['R_sfc_%s' % ct].values
xR['R_sfc'] = (['time','lat','lon'], R_sfc)
xR['R_sfc'].attrs = {'units' : 'W m-2', 'long_name' : 'Total surface reflection (sum of all clear and cloud scenes)'}

xR['Rc_sfc'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['Rc_sfc'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of Rsfc'}
xR['Ra_sfc'] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
xR['Ra_sfc'].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of Rsfc'}

for m in range(1,13):
    Rsfc = xR['R_sfc'].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
    if m ==2: xR['Rc_sfc'][m-1] = np.average(Rsfc[Rsfc.time.dt.month==m],weights=Feb_wts,axis=0)
    else: xR['Rc_sfc'][m-1] = Rsfc[Rsfc.time.dt.month==m].mean(axis=0)
    xR['Ra_sfc'][xR.time.dt.month==m] = xR['R_sfc'][xR.time.dt.month==m].values-xR['Rc_sfc'][m-1].values

#Climatology for S*Aatm and S*Asfc
for loc in ['atm','sfc']:
    xR['S*Ac_%s_clr' % loc] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
    xR['S*Ac_%s_clr' % loc].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of S*A%s' % loc}
    xR['S*Aa_%s_clr' % loc] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
    xR['S*Aa_%s_clr' % loc].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of S*A%s' % loc}

    for m in range(1,13):
        R = xR['S*A_%s_clr' % loc].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
        if m ==2: xR['S*Ac_%s_clr' % loc][m-1] = np.average(R[R.time.dt.month==m],weights=Feb_wts,axis=0)
        else: xR['S*Ac_%s_clr' % loc][m-1] = R[R.time.dt.month==m].mean(axis=0)
        xR['S*Aa_%s_clr' % loc][xR.time.dt.month==m] = xR['S*A_%s_clr' % loc][xR.time.dt.month==m].values-xR['S*Ac_%s_clr' % loc][m-1].values

#Climatology for S*Aclr
xR['S*Ac_clr'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['S*Ac_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of S*Aclr'}
xR['S*Aa_clr'] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
xR['S*Aa_clr'].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of S*Aclr'}

for m in range(1,13):
    Rsfc = xR['S*A_clr'].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
    if m ==2: xR['S*Ac_clr'][m-1] = np.average(Rsfc[Rsfc.time.dt.month==m],weights=Feb_wts,axis=0)
    else: xR['S*Ac_clr'][m-1] = Rsfc[Rsfc.time.dt.month==m].mean(axis=0)
    xR['S*Aa_clr'][xR.time.dt.month==m] = xR['S*A_clr'][xR.time.dt.month==m].values-xR['S*Ac_clr'][m-1].values

#Climatology for Ratm by scene
for ct in ['clr']+cldtypes:

    xR['Rc_atm_%s' % ct] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
    xR['Rc_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of Ratm for %s scenes' % ct}
    xR['Ra_atm_%s' % ct] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
    xR['Ra_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of Ratm for %s scenes' % ct}
    
    if ct != 'clr':
        xR['S*Ac_atm_%s' % ct] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
        xR['S*Ac_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of S*Aatm for %s scenes' % ct}
        xR['S*Aa_atm_%s' % ct] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
        xR['S*Aa_atm_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of S*Aatm for %s scenes' % ct}
        
        xR['Cc_%s' % ct] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
        xR['Cc_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of %s cloud fraction' % ct}
        xR['Ca_%s' % ct] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
        xR['Ca_%s' % ct].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of %s cloud fraction' % ct}

    for m in range(1,13):

        Ratm = xR['R_atm_%s' % ct].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
        if m ==2: xR['Rc_atm_%s' % ct][m-1] = np.average(Ratm[Ratm.time.dt.month==m],weights=Feb_wts,axis=0)
        else: xR['Rc_atm_%s' % ct][m-1] = Ratm[Ratm.time.dt.month==m].mean(axis=0)
        xR['Ra_atm_%s' % ct][xR.time.dt.month==m] = xR['R_atm_%s' % ct][xR.time.dt.month==m].values-xR['Rc_atm_%s' % ct][m-1].values
        
        if ct != 'clr':
            Ratm = xR['S*A_atm_%s' % ct].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
            if m ==2: xR['S*Ac_atm_%s' % ct][m-1] = np.average(Ratm[Ratm.time.dt.month==m],weights=Feb_wts,axis=0)
            else: xR['S*Ac_atm_%s' % ct][m-1] = Ratm[Ratm.time.dt.month==m].mean(axis=0)
            xR['S*Aa_atm_%s' % ct][xR.time.dt.month==m] = xR['S*A_atm_%s' % ct][xR.time.dt.month==m].values-xR['S*Ac_atm_%s' % ct][m-1].values
            
            C = xR['C_%s' % ct].sel(time=slice(np.datetime64('2002-07'),np.datetime64('2022-07')))
            if m ==2: xR['Cc_%s' % ct][m-1] = np.average(C[C.time.dt.month==m],weights=Feb_wts,axis=0)
            else: xR['Cc_%s' % ct][m-1] = C[C.time.dt.month==m].mean(axis=0)
            xR['Ca_%s' % ct][xR.time.dt.month==m] = xR['C_%s' % ct][xR.time.dt.month==m].values-xR['Cc_%s' % ct][m-1].values

        
#
###All-sky reflection and insolation
#
R = np.array(cFlx['toa_sw_all_mon']) #Get R, Ra, and Rc
R[np.isnan(R)] = 0
xR['R'] = (['time','lat','lon'],R)
xR['R'].attrs = {'units' : 'W m-2', 'long_name' : 'All-sky reflection'}
xR['Rc'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['Rc'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of R'}
xR['Ra'] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
xR['Ra'].attrs = {'units' : 'W m-2', 'long_name' : 'Deseasonalized anomaly of R'}

xR['S'] = (['time','lat','lon'],S)
xR['S'].attrs = {'units' : 'W m-2', 'long_name' : 'Insolation'}
xR['Sc'] = (['month','lat','lon'],np.nan*np.ones((12,180,360)))
xR['Sc'].attrs = {'units' : 'W m-2', 'long_name' : 'Climatological (2002-07 to 2022-06) value of S'}
xR['Sa'] = (['time','lat','lon'],np.nan*np.ones(xR['T_all'].shape))
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

    
#
###Save smaller files to simplify storage/transfer
#

#All-sky values
tmp = xr.Dataset()
tmp['time'] = cFlx.time
tmp['lat'] = cFlx.lat
tmp['lon'] = cFlx.lon
tmp['month'] = np.arange(1,13)
tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

for var in ['T_all','T_hat','R','Rc','Ra','R_sfc','Rc_sfc','Ra_sfc','S','Sc','Sa']:
    tmp[var] = xR[var]

filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_all.nc'
os.system('rm %s' % filename)
tmp.to_netcdf(path=filename,mode='w')
    
#Clear-sky values
tmp = xr.Dataset()
tmp['time'] = cFlx.time
tmp['lat'] = cFlx.lat
tmp['lon'] = cFlx.lon
tmp['month'] = np.arange(1,13)
tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

for var in ['T_clr','S*A_clr','R_clr','S*A_atm_clr','R_atm_clr','S*A_sfc_clr','R_sfc_clr','S*Ac_atm_clr','S*Aa_atm_clr','S*Ac_sfc_clr','S*Aa_sfc_clr','S*Ac_clr','S*Aa_clr','Rc_atm_clr','Ra_atm_clr']:
    tmp[var] = xR[var]

filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_clr.nc'
os.system('rm %s' % filename)
tmp.to_netcdf(path=filename,mode='w')

    
#Cloud-type values
for cld in cldtypes:
    print(cld)
    
    tmp = xr.Dataset()
    tmp['time'] = cFlx.time
    tmp['lat'] = cFlx.lat
    tmp['lon'] = cFlx.lon
    tmp['month'] = np.arange(1,13)
    tmp['month'].attrs = {'units' : '1', 'long_name' : 'month of the year'}

    for var in ['T_','C_','R_atm_','S*A_atm_','R_sfc_','Rc_atm_','Ra_atm_','S*Ac_atm_','S*Aa_atm_','Cc_','Ca_']:
        tmp[var+cld] = xR[var+cld]

    filename = dir_data+'CERES/FluxByCldType/FBCT_decomposition_%s.nc' % cld
    os.system('rm %s' % filename)
    tmp.to_netcdf(path=filename,mode='w')
    
    

