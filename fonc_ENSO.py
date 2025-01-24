#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:36:25 2023

@author: parouffe

fonction to detrend
fonction regression multilineaire using curve fit
"""


from netCDF4 import Dataset
import numpy as np
import pandas as pd
import xarray as xr
from datetime import date
from scipy.optimize import curve_fit
from scipy import signal


#%%
def detrend_anom_1D(var):
    var_dt = signal.detrend(var)
    var_climato = np.zeros((12))
    nt = len(var)
    ny = int(nt/12)
    for k in range(12):
        p = np.arange(k,nt,12)
        var_climato[k] = var_dt[p].mean(axis=0)
        
    var_anom_dt = var_dt - np.tile(var_climato,(ny))
    return var_anom_dt

    
#%%
def RML(E_t, C_t, var_dt, nlat, nlon):

    def func(X, alpha_xy, beta_xy):
        E_t,C_t = X 
        return E_t*alpha_xy + C_t*beta_xy
    
    alpha_xy = np.zeros((nlat,nlon))
    beta_xy = np.zeros((nlat,nlon))
    p_err = np.zeros((nlat,nlon))
    
    my_arr = np.ma.array([E_t,C_t]).transpose()
    X = pd.DataFrame(my_arr,columns=['E','C'])

    for i in range(nlat):
        for j in range(nlon):
            # prepare
            x0 = E_t    #E_xy.sel(lat=Z_opaque.lat[i], lon=Z_opaque.lon[j])
            x1 = C_t    #C_xy.sel(lat=Z_opaque.lat[i], lon=Z_opaque.lon[j])
            
            # y_data = data_anom_sans_trend[:,i,j]
            y_data = var_dt[:,i,j]
            y_data[np.isnan(y_data)]=0
            [alpha, beta], pcov = curve_fit(func, (x0,x1), y_data)
            #print(Z_opaque.lat[i].values, Z_opaque.lon[j].values, alpha, '--', beta)
            alpha_xy[i,j] = alpha
            beta_xy[i,j] = beta
            p_err[i,j] = np.sqrt(np.diag(pcov))[0]
    # mask = np.ma.getmask(data_anom_sans_trend[0,:,:])            
    mask = np.ma.getmask(var_dt[0,:,:])
    alpha_xy = np.ma.masked_where(mask==True,alpha_xy)
    beta_xy = np.ma.masked_where(mask==True,beta_xy)
    p_err[np.isinf(p_err)]=np.nan
    
    return alpha_xy, beta_xy, p_err
#%% detrend anomalies

def detrend_anom (var):
    nt,nlat,nlon = np.shape(var)
    datenum = np.arange(0,nt,1)

    # detrend
    var_dt = signal.detrend(var,0)

    # climato
    var_climato = np.zeros((12,nlat,nlon))
    ny = int(nt/12)
    for i in range(12):
        p = np.arange(i,nt,12)
        var_climato[i,:,:] = var_dt[p,:,:].mean(axis=0)
        
    var_anom_dt = var_dt - np.tile(var_climato,(ny,1,1))
    var_anom_dt = np.ma.masked_where(var==True,var_anom_dt)
    
    return var_anom_dt

#%%

def PCs (ssta):
    """ 2) reorganise data : sst = [time, nlat*nlon] """

        
    """ 3) covariance matrix """
    # spatial covariance matrix : nlat*nlon times series
    covX = np.dot(ssta,np.transpose(ssta)) # unit = °C^2
    
    """ 4) Eigen vectors and eigen values of covariance matrix """
    from numpy import linalg
    eig_val, eig_vec = linalg.eig(covX) # dimensionless
    # already sorted by increasing order of eigen values

    """ 5) explained variance  (ev) """
    trace = eig_val.sum()
    ve = eig_val/trace #*100
        
    """ 6) PCs here = eigen vectors """
    # pourquoi on divise par la std de eig_vec ?
    pc1 = eig_vec[:,0]/np.nanstd((eig_vec[:,0])) # unit : "s.d."
    pc2 = eig_vec[:,1]/np.nanstd((eig_vec[:,1]))
    
    return pc1, pc2, eig_vec[:,:2], ve[:2]
    
#%% 

def EOF (pc1, pc2, data_np, eig_vec1, eig_vec2):
    E = (pc1-pc2)/np.sqrt(2.) # dimensionless
    C = (pc1+pc2)/np.sqrt(2.)
    
    """ 7) EOFs """
    # EOFs = np.dot(np.transpose(data_np),eig_vec)
    # EOFs = np.dot(np.transpose(eig_vec),data_np)
    eof1 =np.dot(np.transpose(data_np),eig_vec1)
    eof2 =np.dot(np.transpose(data_np),eig_vec2)

    # note: the eof is multiplied by the standard deviation of the PCs because the 
    # projection of the data on the the PC is in °C per s.d. change in the PC
    eof1 = np.reshape(eof1,(nlat,nlon))*np.nanstd(eig_vec1) # °C x s.d. / s.d. = °C
    eof2 = np.reshape(eof2,(nlat,nlon))*np.nanstd(eig_vec2)
    
    """ 8) E and C pattern """
    data = np.reshape(data_np,(nt,nlat,nlon))

    alpha, beta, p = RML(E, C, data, nlat, nlon)
    
    return pc1, pc2, E, C, eof2, eof2, alpha, beta, ve[0], ve[1]


def func(X, alpha_xy, beta_xy):
    E_t,C_t = X 
    return E_t*alpha_xy + C_t*beta_xy

def calc_ab():
	""" code """
	depths1=[0,50,100,150]
	depths2=[50,100,150,200]
	for sim in range(1):#len(files)
		
		run = files[sim][-6:-3] 
		nc = Dataset(files[sim])

		for d in range(len(depths1)):
		    
		    lvl1 = np.abs(zt-depths1[d]).argmin()
		    lvl2 = np.abs(zt-depths2[d]).argmin()  
		    """ detrend """
		    var = nc.variables[variable][:,lvl1:lvl2,:,-40:]
		    
		    var = np.ma.masked_where(var==np.nan,var)
		    
		    var = np.ma.mean(var,1)
		    nt, nlat, nlon = np.shape(var)
		    
		    var_dt = signal.detrend(var,0)
		
		    var_climato = np.zeros((12,nlat,nlon))
		    ny = int(nt/12)
		    for i in range(12):
		        p = np.arange(i,ny+11,12)
		        var_climato[i,:,:] = var_dt[p,:,:].mean(axis=0)
		    
		    var_anom_dt = var_dt - np.tile(var_climato,(ny,1,1))
		    var_anom_dt = np.ma.masked_where(var==True,var_anom_dt)
		    var_anom_dt = var_anom_dt/np.nanstd(var_anom_dt,0,keepdims=True)
		
		    """ get indices E and C """
		    f = Dataset(f_indice)
		    E_t = f.variables['E'][sim,:]
		    C_t = f.variables['C'][sim,:]
		    f.close()
		    
		    """ calculate RML """
		    nt, nlat, nlon = np.shape(var_anom_dt)
		    
		    alpha, beta, p = RML(E_t, C_t, var_anom_dt, nlat, nlon)
		    
		    alpha = np.ma.masked_where(var[0,:,:]==True,alpha)
		    beta = np.ma.masked_where(var[0,:,:]==True,beta)
    return alpha, beta
