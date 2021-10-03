#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# I.S. Giddy - tools for MIZ export paper
# Everything pertaining to quantifyinng export fluxes from gliders

# A few functions for fitting Martins Curve
from scipy.optimize import curve_fit

def martins_curve(Z,fluxref,b,Zref=55):
    return fluxref*(Z/Zref)**-b

def martins_curve_lin(Z,flux100,b):
    return -b*Z +(flux100) 

def martins_curve_log(xdata,ydata,Zref=55):
    """Fit data to martins curve (Martin et al., 1987) power law with weights according to a log scale"""
    from scipy.optimize import curve_fit
    import numpy as np
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log(xdata/Zref)
    # Apply fscaley
    ydata_log = np.log(ydata)
    # Fit linear martins curve
    popt_log, pcov_log = curve_fit(martins_curve_lin, xdata_log, ydata_log)
    ydatafit_log = np.exp(martins_curve_lin(xdata_log, *popt_log))

    return (popt_log, pcov_log, ydatafit_log)

def fit_export_events(spike_dataset, glider_id,plot_xlim_min=None,plot_xlim_max=None,moving_window=10, min_depth = 70,max_depth=360,r2_threshold=0.2,output_figure=True,cbar_label='Backscatter spikes (m$^{-1}$)',sci_notation=True,levels=None):
    """
    Iterates through backscatter spikes to identify export events
    Expects a pandas dataframe that has been binned to 2 days in the horizontal 
    Could be improved :p 
    Still needs a unit test but this feels complex to write (a reflection of the code hack)
    
    
    """
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import xarray as xr
    

    
    #find index of maximum depth to fit regression
    max_depth_index = np.abs(spike_dataset.index-max_depth).argmin()
    min_depth_index = np.abs(spike_dataset.index-min_depth).argmin()

    dof=pd.Series(spike_dataset.columns).dt.dayofyear # convert datetimes to day of year 
    lst_dict = []
    regress_dict = []
    for t in range(1,len(spike_dataset.columns),1): 
        ind=[]
        val=[]
        for i in (spike_dataset.index[min_depth_index:max_depth_index]): 
            # selects period over which to compute the sinking rate # a ten day period appears to be the best choice
            # compute index and value of that depth levels maximum backscatter spike
            ind.append(spike_dataset.iloc[:,t:t+moving_window].loc[i,:].argmax())
            val.append((spike_dataset.iloc[:,t:t+moving_window].loc[i,:]).max())

        ind=np.array(ind)
        val=np.array(val)

        dof = ((((spike_dataset.columns[t:t+moving_window]).dayofyear+ 180) %365)-180)
        year = pd.DatetimeIndex(spike_dataset.columns[t:t+moving_window]).year
        time_max = dof[ind]
        # check if there is an export event
#         if there is compute the regression, if not skip one timestep forward
        if ((np.nanmean(val[4:6]) > np.nanmean(val[-15:-1]))& #&(np.nanmean(val[:4]) > val[np.int(len(val)/2)])&\
        ((np.nanmean(time_max[0:2]))<np.nanmean(time_max[-3:-1]))): # 

            
            for a in range(len(val)-1):
                if (val[a]-val[a+1]<0):
                    stop = val[val[a]-val[a+1]<0].argmin()
            # #####
            #######
            
            X = np.array(spike_dataset.index[min_depth_index:max_depth_index])
            Y = np.array(time_max)
            mask = ~np.isnan(X)&~np.isnan(Y)

            X= X[mask].reshape(-1,1)
            Y =Y[mask].reshape(-1,1)
            linear_regressor = LinearRegression()

            linear_regressor.fit(X,Y)
            Y_pred = linear_regressor.predict(X)

            slope=linear_regressor.coef_
            intercept = linear_regressor.intercept_         
            r2=linear_regressor.score(X,Y)
            if r2>r2_threshold:
                year1 = year[0]
                date = (Y_pred)[0]
                d = X.squeeze()
                tm = (Y_pred).squeeze()
                vals = [t,year1,date,slope,intercept,r2]
                cols = ['ind','year','doy','slope','intercept','r2']
                lst_dict.append(dict(zip(cols, vals)))
                colsR = ['depth','max_spike','doy','year']
                valsR = [d,val,tm,year1.repeat(len(d))]
                regress_dict.append(dict(zip(colsR,valsR)))
            
    df = pd.DataFrame(lst_dict)
    dfR = pd.DataFrame(regress_dict)

    # compute sinking rate by finding the gradient of the fitted regression line (depth/days)
    sinking_rate=[]
    for row in dfR.index:
        dnum=((np.abs(dfR.iloc[row].doy[-1]-dfR.iloc[row].doy[0])))
        depth_diff=(dfR.iloc[row].depth.max()-dfR.iloc[row].depth.min())
        sinking_rate.append(depth_diff/dnum)
    df['sinking_rate']=sinking_rate
    
    if output_figure==True:
        import cmocean.cm as cmo
        import matplotlib.pyplot as plt
        
        import matplotlib 
        matplotlib.rc('xtick', labelsize=12) 
        matplotlib.rc('ytick', labelsize=12) 
        matplotlib.rcParams.update({'font.size': 14})
        
        
        days = (((pd.Series(spike_dataset.columns).dt.dayofyear+ 180) %365)-180)

        fig,ax=plt.subplots(1,1,figsize=[15,5])
        xx,yy=np.meshgrid(days,spike_dataset.index)
        levels=levels
        cs=ax.contourf(xx,yy,spike_dataset,levels=levels,cmap=cmo.delta)
        for row in dfR.index:  #fit regression lines
            ax.plot(dfR.iloc[row].doy,dfR.iloc[row].depth,c='r')
        ax.set_ylim(spike_dataset.index[-1],spike_dataset.index[0],10)
        cbar=plt.colorbar(cs)
        cbar.set_label(cbar_label)

        ax.set_xlabel('Day of year')
        ax.set_ylabel('Depth (m)')
        ax.set_xlim(plot_xlim_min,plot_xlim_max)
        
        if sci_notation==True:
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()

#         ax1.set_title('SG2018')

        plt.savefig('export_events_{}.png'.format(glider_id),bboax_inches='tight')

        return df, dfR, fig#.get_gca()
    else:
        return df, dfR
    
    
