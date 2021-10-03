#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# I. Giddy - tools for MIZ export paper
# Work In Progress
# Code adapted from Glider Tools, Oregan State PP website and Lionel Arteagas recent work

import numpy as np
    
def sunset_sunrise(time, lat, lon):
    """
    Calculates the local sunrise/sunset of the glider location.

    The function uses the Skyfield package to calculate the sunrise and sunset
    times using the date, latitude and longitude. The times are returned
    rather than day or night indices, as it is more flexible for the quenching
    correction.


    Parameters
    ----------
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    lat: numpy.ndarray or pandas.Series
        The latitude of the glider position.
    lon: numpy.ndarray or pandas.Series
        The longitude of the glider position.

    Returns
    -------
    sunrise: numpy.ndarray
        An array of the sunrise times.
    sunset: numpy.ndarray
        An array of the sunset times.

    """
    

    from pandas import DataFrame
    import numpy as np
    import pandas as pd
    from skyfield import api

    ts = api.load.timescale()
    eph = api.load("de421.bsp")
    from skyfield import almanac

    df = DataFrame.from_dict(dict([("time", time), ("lat", lat), ("lon", lon)]))

    # set days as index
    df = df.set_index(df.time.values.astype("datetime64[D]"))

    # groupby days and find sunrise/sunset for unique days
    grp_avg = df.groupby(df.index).mean()
    date = grp_avg.index.to_pydatetime()
    date = grp_avg.index

    time_utc = ts.utc(date.year, date.month, date.day, date.hour)
    time_utc_offset = ts.utc(
        date.year, date.month, date.day + 1, date.hour
    )  # add one day for each unique day to compute sunrise and sunset pairs

    bluffton = []
    for i in range(len(grp_avg.lat)):
        bluffton.append(api.wgs84.latlon(grp_avg.lat[i], grp_avg.lon[i]))
    bluffton = np.array(bluffton)

    sunrise = []
    sunset = []
    for n in range(len(bluffton)):

        f = almanac.sunrise_sunset(eph, bluffton[n])
        t, y = almanac.find_discrete(time_utc[n], time_utc_offset[n], f)

        if not t:
            if f(time_utc[n]):  # polar day
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 0, 1
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 23, 59
                    ).to_datetime64()
                )
            else:  # polar night
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 11, 59
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 12, 1
                    ).to_datetime64()
                )

        else:
            sr = t[y == 1]  # y=1 sunrise
            sn = t[y == 0]  # y=0 sunset

            sunup = pd.to_datetime(sr.utc_iso()).tz_localize(None)
            sundown = pd.to_datetime(sn.utc_iso()).tz_localize(None)

            # this doesnt look very efficient at the moment, but I was having issues with getting the datetime64
            # to be compatible with the above code to handle polar day and polar night

            su = pd.Timestamp(
                sunup.year[0],
                sunup.month[0],
                sunup.day[0],
                sunup.hour[0],
                sunup.minute[0],
            ).to_datetime64()

            sd = pd.Timestamp(
                sundown.year[0],
                sundown.month[0],
                sundown.day[0],
                sundown.hour[0],
                sundown.minute[0],
            ).to_datetime64()

            sunrise.append(su)
            sunset.append(sd)

    sunrise = np.array(sunrise).squeeze()
    sunset = np.array(sunset).squeeze()

    grp_avg["sunrise"] = sunrise
    grp_avg["sunset"] = sunset

    # reindex days to original dataframe as night
    df_reidx = grp_avg.reindex(df.index)
    sunrise, sunset = df_reidx[["sunrise", "sunset"]].values.T

    return sunrise, sunset
    

def AustinPetzold_1986(Lambda,k490):
    """
    Created on Thu Jun  4 12:08:23 2020
    @author: Lionel Alejandro Arteaga Quintero (laaq) (laaq@princeton.edu, lionel.arteagaquintero@nasa.gov, artionel@gmail.com)
    Given a reference k490 vlaue, determine k(lambda) for a specified lambda.
    Taken from the CbPM code of Westberry et al (2008) as 
    in http://www.science.oregonstate.edu/ocean.productivity/cbpm2.code.php
    Based on Austin and Petzold (1986).
    """
    wave = [350, 360, 370, 380, 390, 400, 410, 420\
            , 430, 440, 450, 460, 470, 480, 490, 500, 510\
            , 520, 530, 540, 550, 560, 570, 580, 590, 600\
            , 610, 620, 630, 640, 650, 660, 670, 680, 690, 700];

    M = [2.1442, 2.0504, 1.9610, 1.8772, 1.8009, 1.7383, 1.7591,\
         1.6974, 1.6108, 1.5169, 1.4158, 1.3077, 1.1982, 1.0955,\
         1.0000, 0.9118, 0.8310, 0.7578, 0.6924, 0.6350, 0.5860,\
         0.5457, 0.5146, 0.4935, 0.4840, 0.4903, 0.5090, 0.5380,\
         0.6231, 0.7001, 0.7300, 0.7301, 0.7008, 0.6245, 0.4901, 0.2891];

    Kdw = [0.0510, 0.0405, 0.0331, 0.0278, 0.0242, 0.0217, 0.0200,\
           0.0189, 0.0182, 0.0178, 0.0176, 0.0176, 0.0179, 0.0193, 0.0224,\
           0.0280, 0.0369, 0.0498, 0.0526, 0.0577, 0.0640, 0.0723, 0.0842,\
           0.1065, 0.1578, 0.2409, 0.2892, 0.3124, 0.3296, 0.3290, 0.3559,\
           0.4105, 0.4278, 0.4521, 0.5116, 0.6514];

    # Interpolate to wavelength of interest
    for i in range(len(wave)):
        if wave[i] >= Lambda:
            l1 = wave[i]
            k1 = Kdw[i]
            m1 = M[i]
            l0 = wave[i-1]
            k0 = Kdw[i-1]
            m0 = M[i-1]
            break

    num = Lambda - l0
    den = l1 - l0
    frac = num / den
    
    kdiff = k1 - k0
    Kdw_l = k0 + frac * kdiff

    mdiff = m1 - m0
    M_l = m0 + frac*mdiff

    # Get reference wavelength (490 for now) and appply model
    ref = 14
    Kd = (M_l/M[ref]) * (k490 - Kdw[ref]) + Kdw_l
    return Kd

def time_average_per_dive(dives, time):
    """
    Gets the average time stamp per dive. This is used to create psuedo
    discrete time steps per dive for plotting data (using time as x-axis
    variable).

    Parameters
    ----------
    dives : np.array, dtype=float, shape=[n, ]
        discrete dive numbers (down = d.0; up = d.5) that matches time length
    time : np.array, dtype=datetime64, shape=[n, ]
        time stamp for each observed measurement

    Returns
    -------
    time_average_per_dive : np.array, dtype=datetime64, shape=[n, ]
        each dive will have the average time stamp of that dive. Can be used
        for plotting where time_average_per_dive is set as the x-axis.
    """
    from numpy import array, datetime64, nanmean
    from pandas import Series

    atime = array(time)
    dives = array(dives)
    if isinstance(atime[0], datetime64):
        t = atime.astype("datetime64[s]").astype(float)
    else:
        t = atime

    t_grp = Series(t).groupby(dives)
    t_mid = nanmean([t_grp.max(), t_grp.min()], axis=0)
    t_ser = Series(t_mid, index=t_grp.mean().index.values)
    diveavg = t_ser.reindex(index=dives).values
    diveavg = diveavg.astype("datetime64[s]")

#     diveavg = transfer_nc_attrs(getframe(), time, diveavg, "_diveavg")

    return diveavg


def mixed_layer_depth(
    dives, depth, dens_or_temp, thresh=0.01, ref_depth=10, return_as_mask=False
    ):
    """
    Taken from GliderTools.
    Calculates the MLD for ungridded glider array. 
    
    You can provide density or temperature.
    The default threshold is set for density (0.01).

    Parameters
    ----------
    dens_or_temp : array, dtype=float, shape=[n, ]
        temperature/density of the entire dataset
    depth : array, dtype=float, shape=[n, ]
        depth for each measurement
    dives : array, dtype=float, shape=[n, ]
        will be used to calculate MLD per dive
    thresh : float=0.01
        threshold for difference
    ref_depth : float=10
        reference depth for difference
    return_as_mask : bool=False
        sets output to be a mask or an array of depth values

    Return
    ------
    mld : array
        will be a mask of shape=[n, ] or an array of depths the length of the
        number of unique dives.
    """


    import numpy as np
    from pandas import DataFrame

    def mld_profile(dens_or_temp, depth, thresh, ref_depth, mask=False):

        i = np.nanargmin(np.abs(depth - ref_depth))

        if np.isnan(dens_or_temp[i]):
            mld = np.nan
        else:
            dd = dens_or_temp - dens_or_temp[i]  # density difference
            dd[depth < ref_depth] = np.nan
            abs_dd = abs(dd - thresh)
            depth_idx = np.nanargmin(abs_dd)
            mld = depth[depth_idx]

        if mask:
            return depth <= mld
        else:
            return mld

    arr = np.c_[dens_or_temp, depth, dives]
    col = ['dens', 'depth', 'dives']
    df = DataFrame(data=arr, columns=col)

    grp = df.groupby('dives')
    mld = grp.apply(
        lambda g: mld_profile(
            g.dens.values,
            g.depth.values,
            thresh,
            ref_depth,
            mask=return_as_mask,
        )
    )

    if return_as_mask:
        return np.concatenate([l for l in mld])
    else:
        return mld

def pandas_grid(var,dives,depth,bins=np.arange(0,1000,2)):
    """
    A GliderTools derivative.
    Inputs variable, dives and depths of equal length and a predefined bin size
    Returns a pandas dataframe
    """
    # grid data to per profile

    import numpy as np
    from pandas import cut, Series
    from xarray import DataArray
    import numpy as np
    from numpy import array, c_, unique, diff

    xvar, yvar = dives.copy(), depth.copy()
    z = Series(var)
    y = array(depth)
    x = array(dives)

    u = unique(x).size
    s = x.size
    if (u / s) > 0.2:
        raise UserWarning(
            'The x input array must be psuedo discrete (dives or dive_time). '
            '{:.0f}% of x is unique (max 20% unique)'.format(u / s * 100)
        )

    chunk_depth = 50

    

    labels = c_[bins[:-1], bins[1:]].mean(axis=1)
    bins = cut(y, bins, labels=labels)

    grp = Series(z).groupby([x, bins])

    how='mean'
    grp_agg = getattr(grp, how)()
    gridded = grp_agg.unstack(level=0)
    gridded = gridded.reindex(labels.astype(float))
    
    return gridded


def cbpm_bgcfloats(chl_z,Cphyto_z,irr,daylength):
    """
    Computes CbPM model, adjusted from L. Arteaga - see explanation below
    
    This algorithm requires gridded chl-a, phytoplankton carbon, irradiance and daylength
     - you can using the pandas_grid function found in this file
     
    I then suggest implementing as in the example below: 
    
    I first computed phytoplankton carbon from backscatter using
    compute_cphyto (which is based on Behrenfeld relationship)
    and daylength using the sunset_sunrise function in this file
    and subtracting the two to get a daylength in hours
 
    pp_z_sat643=np.ndarray([200,len(pars)])  # output Primary Production
    mu_z_sat=np.ndarray([200,len(pars)])  # output  Phytoplankton growth rate
    par_z_sat=np.ndarray([200,len(pars)]) # output Photosynthetically available radiation 
    prcnt_z_sat=np.ndarray([200,len(pars)]) #output % remaining irradiance

    for i in range(len(length_of_timeseries)):
        pp_z_sat643[:,i],mu_z_sat[:,i],par_z_sat[:,i],prcnt_z_sat[:,i]   = ppt.cbpm_bgcfloats(chl_z.iloc[:,i].values,
                                                                            cphyto_z.iloc[:,i].values,
                                                                            sat_PAR.iloc[i],
                                                                                dl.iloc[i])
    
    You can then integrate in depth to get depth integrated PP. 
    
    
    Created on Wed Jun  3 11:19:55 2020
    @author: Lionel Alejandro Arteaga Quintero (laaq) (laaq@princeton.edu, lionel.arteagaquintero@nasa.gov, artionel@gmail.com)
    CODE DESCRIPTION: Adapation of the Carbon-based Productivity Model (CbPM) by 
    Westberry, et al. (2008) to estimate depth-resolved phytoplankton growth variables
    forced by output from biogeochemical (bgc) profiling floats (adapted specifically for 
    SOCCOM float data (https://soccom.princeton.edu)).
    Code adapted from the satellite-based CbPM "updated" model at the OSU Ocean 
    Productivity web site (http://sites.science.oregonstate.edu/ocean.productivity/cbpm2.code.php)
    GENERAL MODEL (CbPM) DESCRIPTION: This is a spectrally resolved version of the cbpm, 
    using nine separate wavelengths.  It is also depth resolved, integrating the effects 
    from the surface down to a fixed depth of 200 m.
    
    The CbPM algorithm estimates productivity using chl (m-1), bbp (m-1), surface 
    irradiance (Einsteins m-2 d-1), k490 (m-1), mld (m), zno3 (m) and day length (hours).
    Net primary productivity is phytoplankton carbon (Cphyto) \times growth rate, where 
    carbon is  proportional to particulate backscatter (bbp). The orginal satellite-based 
    CbPM converts satellite estimates of bbp to Cphyto. Here we used estimates
    of Cphyto obtained from BGC-SOCCOM floats based on particulate 
    organic carbon (POC) estimated from float measaurements of bbp.
        Cphyto (mg m^-3) = 0.19 * POC (mg m^-3) + 8.7 (Graff et al, 2015, Deep-Research I)
        (\mug l^-1) = (mg m^-3)
    Phytoplankton growth rate is a function of nutrient and temperature stresst (f(nut,t))
    and photoacclimation (f(Ig))
        growth rate (u) = umax * f(nut,T) * f(Ig)

    where:    
        umax = 2
        f(nut,T) = ((Chl/C(z)) - (Chl/C)mu=0) / ((Chl/C)max - (Chl/C)mu=0)
        f(Ig) = 1 - exp (-5 * Ig)

    and:
        (Chl/C(z)) = Ratio of float-basee chl and carbon at each depth (z)
        (Chl/C)max = 0.022 + (0.045-0.022) * exp (-3 * Ig) (max Chl/C given Ig)
        (Chl/C)mu=0 = (minimum) Chl:C at growth rate (mu) = 0    
    The above items are analyzed for nine separate wavelengths, and is vertically 
    resolved to a depth of 200 m.

    For more details, please see the paper by Westberry, et al (2008).    

    """
    
    import numpy as np
    import pandas as pd
    
    # Spectral variables
    Lambda = [ 400, 412, 443, 490, 510, 555, 625, 670, 700];
    parFraction = [  0.0029, 0.0032, 0.0035, 0.0037, 0.0037, 0.0036, 0.0032, 0.0030, 0.0024];
    X = [ .11748, .122858, .107212, .07242, .05943, .03996, .04000, .05150, .03000];
    e = [ .64358, .653270, .673358, .68955, .68567, .64204, .64700, .69500, .60000];
    Kw = [ .01042, .007932, .009480, .01660, .03385, .06053, .28400, .43946, .62438];

    # Initialization variables for testing
    #year = 2015
    #month = 4
    #day = 24
    #lat = 30
    #chl_ml = 1 
    #irr = 30

    # Initialize necessary values for the model
    y0 = 0.0003 # min Chl:C ratio when mu = 0
    umax = 2.0 #after Banse (1991)
    chlC_z = np.zeros(200) # Phytoplankton Chl:C ratio (mg/mg)
    chlC_z[:] = np.nan
    nutTempFunc_z = np.zeros(200) # Nutrient dependent term
    nutTempFunc_z[:] = np.nan
    IgFunc_z = np.zeros(200) # Light dependent term
    IgFunc_z[:] = np.nan
    mu_z = np.zeros(200) # Phytoplankton growth rate
    mu_z[:] = np.nan
    prcnt_z = np.zeros(200) # Remaining light fraction
    prcnt_z[:] = np.nan
    pp_z = np.zeros(200) # Primary production
    pp_z[:] = np.nan
    Ezlambda = np.zeros([200,9]) # Fraction of light at nine wavelengths
    Ezlambda[:] = np.nan
    par_z = np.zeros(200) # Photosynthetically available radiation 
    par_z[:] = np.nan

    # Attenuation coefficient at 490nm based on Equation 8 of 
    # Morel, et al. (2007, Remote Sens. Environ.)
    chl_surf = chl_z[0]
    k490 = 0.0166 + 0.0773 * pow(chl_surf,0.6715);

    # Multispectral comonent of the model
    klambda = np.zeros(len(Lambda))
    klambda[:] = np.nan
    E0 = np.zeros(len(Lambda))
    E0[:] = np.nan
    kbio = np.zeros(len(Lambda))
    kbio[:] = np.nan
    kd = np.zeros(len(Lambda))
    kd[:] = np.nan
    kdif = np.zeros(len(Lambda))
    kdif[:] = np.nan

    for n in range(len(Lambda)):
        klambda[n] = AustinPetzold_1986(Lambda[n],k490)
        E0[n] = irr * parFraction[n]
        #Ez_mld(i) = Eo(i) .* 0.975 .* exp(-klambda(i) .* mld/2.0);
    # Calculate Kd offset carry through to depth non-chl attenuation
    #for n in range(len(Lambda)):
    #   kbio = X[n] * pow(chl_ml,e[n])
    #  kd[n] = Kw[n] + kbio
    # kdif[n] = klambda[n] - kd[n]
    for z in range(len(chl_z)):
        if z == 0:        
            for n in range(len(Lambda)):
                Ezlambda[z,n] = E0[n] * 0.975 * np.exp(-klambda[n] * z)
        else:
            for n in range(len(Lambda)):
                kbio = X[n] * pow(chl_z[z-1],e[n]);   # after Morel and Maritorena (2001)
                kd[n] = Kw[n] + kbio
                #kdif[n] = klambda[n] - kd[n]
                #kd[n] = kdif[n] + kd[n];
                Ezlambda[z,n] = Ezlambda[z-1,n] * np.exp(-kd[n] * 1)
        par_z[z] = 0.0;
        for n in range(len(Lambda)-1):
            par_z[z] = ((Lambda[n+1] - Lambda[n]) * (Ezlambda[z,n+1] + Ezlambda[z,n]) /2) + par_z[z]
        chlC_z[z] = chl_z[z] / Cphyto_z[z]
        chlCarbonMax_z = 0.022 + (0.045-0.022) * np.exp(-3.0 * par_z[z] / np.nanmean(daylength))
        nutTempFunc_z[z] = (chlC_z[z] - y0) / (chlCarbonMax_z - y0)
        
        if nutTempFunc_z[z] > 1: 
            nutTempFunc_z[z] = 1
        IgFunc_z[z] = 1 - np.exp(-5.0 * par_z[z] / np.nanmean(daylength)) 
        mu_z[z] = umax * nutTempFunc_z[z] * IgFunc_z[z]
        if mu_z[z] > umax:
            mu_z[z] = umax
        prcnt_z[z] = par_z[z] / irr * 0.975

        # Track 1% of surf. irradiance
        if prcnt_z[z] >= 0.01:
            mzeu = z
        
        # Depth-resolved primary production
        pp_z[z] = mu_z[z] * Cphyto_z[z]
        
    return pp_z,mu_z,par_z,prcnt_z   


def compute_cphyto(bbp):
    """
    Converts backscatter to phytoplankton carbon
    Based on Behrenfeld
    """
    
    import numpy as np
    
    
    if np.any(bbp) < 0.00035:        
        bbp = 0.00036
    cphyto = 13000 * (bbp - 0.00035)
    cphyto = 35400 * (bbp - 0.00035)
    return cphyto

def platt(chl,par,kd,daylength):
    
    """
    Simple implementation of the PLATT model
    """
    # PLATT

    global_alpha_slope = 0.649                
    global_alpha_exponent =  0.865
    global_pmax_slope = 49.934
    global_pmax_exponent =  0.890


    alpha = global_alpha_slope * (chl**global_alpha_exponent) 
    pmax = global_pmax_slope * (chl**global_pmax_exponent)


    pars=par*((60*60*daylength)/1e6)# converts to Einsteins per day
    midday_irradiance = par
    adaptation_parameter = pmax/alpha

    p = pmax*(1-np.exp(-midday_irradiance/adaptation_parameter)) 

    ek = ((pmax/alpha) * 60 * 60 )/ 1e6   
    im = par / ek

    irradiance_daily = midday_irradiance/6

    dimensionless_irradiance_daily = irradiance_daily/adaptation_parameter

    fim = 0.7576 * np.log(im) + 0.5256    # dimensionless function for I* solved analytically by platt 1980

    pp = (p * (fim / (-kd)))  # to compute PP integrated over the entire water column assuming vertically uniform chl
    
    return pp

def calculate_chl_tot(chl):
    
    if (chl < 1):
        chl_tot = 38 * (chl**0.425)
    else:
        chl_tot = 40.2 * (chl**0.507)
    return chl_tot

def calculate_zeu(chl_tot):
    """
    Computes euphotic layer depth based on total chlorophyll
    """
    if chl_tot <=0:
        z_eu = NaN
    else:       
        z_eu = 200.0 * chl_tot**(-0.293)       
    if (z_eu <= 102.0):
        z_eu = 568.2 * chl_tot**(-0.746)    
    return z_eu

def vgpm(chl,par,sst,daylength,version='eppley'):
    """
    implementation of standard or Eppley primary productivity model
    """ 
    if version == 'eppley':
    
        pb_opt = 1.54 *10**((0.0275*sst) - 0.07)
    else: 
    #        /* Calculate the Pb_opt from satellite sea surface temperature (sst).    */  # This makes the algorithm standard VGPM
        if (sst < -10.0): 
            pb_opt = 0.00 
        elif (sst <  -1.0):
            pb_opt = 1.13
        elif (sst >  28.5):
            pb_opt = 4.00      
        else:
            pb_opt = (1.2956 + 2.749e-1*sst + 6.17e-2*sst**2 - 2.05e-2*sst**3
           + 2.462e-3*sst**4 - 1.348e-4*sst**5 + 3.4132e-6*sst**6 - 3.27e-8*sst**7)

    chl_tot = calculate_chl_tot(chl)
    z_eu  = calculate_zeu(chl_tot) 
    f_par = 0.66125 * par / ( par + 4.1 )
#     volume_function = f_par * z_eu 
    NPP = chl * pb_opt * daylength * f_par * z_eu 
    return NPP

def calculate_cbpm(bbp, par, chl, kd, dl, mld):
    
    """
    Carbon Based Productivity Model
    First Described by Behrenfeld et al 2005, and then updated by Westberry et al 2008
    
    CbPM relates NPP to phytoplankton carbon biomass (C_phyto) and growth rate (u). 
    The approach is made possible by two recent developments: (1) the observation that total particulate carbon concentration 
    and C_phyto covary with light scattering properties ( Loisel et al. 2001, Stramski et al. 1999, DuRand and Olsen 1996, Green et al. 2003,
    Green and Sosik 2004, Behrenfeld and Boss 2003, 2006b) and (2) the construction and application of spectral matching algorithms to satellite 
    data for simultaneously retrieving information on particulate backscattering scattering coefficients, phytoplankton pigment absorption, 
    and colored dissolved organic carbon absorption ( Garver and Siegel,1997; Maritorena et al., 2002; Siegel et al., 2002). 
    
    Input: Backscatter
           Par
           Chl - surface chlorophyll (upper 20 m)
           kd - light attenuation coefficient
           dl - daylength
           mld
    """    

    if np.any(par) <= 0:
        npp = 0
        
    umax = 2
    
    #if np.any(bbp) < 0.00035:
    if bbp < 0.00035:
        bbp = 0.00036
    cphyto = 13000 * (bbp - 0.00035)
    cphyto = 35400 * (bbp - 0.00035)

    
    ig = par / dl * np.exp(-kd * (mld / 2))
    fig = 1 - np.exp(-3 * ig)
    chl_c_max = 0.022 + (0.045 - 0.022) * np.exp(-3 * ig)
    chl_c = chl / cphyto
#     fnt = (chl_c - 0.0003) / (chl_c_max - 0.0003)
    fnt = chl_c / chl_c_max
    irrfunc = 0.66125 * par / (par + 4.1)
    growth = umax * fnt * fig
    chl_tot = calculate_chl_tot(chl)   
    zeu  = calculate_zeu(chl_tot)   # dont need kd but still need PAR?? 
#     zeu = -np.log(0.01) / -kd   # remove negative   #kd slope from fitted par profile
    NPP = cphyto * growth * irrfunc * zeu
    #npp = nansum(npp)
    return NPP


def photic_depth(par, dives, depth, return_mask=False, ref_percentage=1):
    """
    Algebraically calculates the euphotic depth.

    The function calculates the euphotic depth and attenuation coefficient (Kd)
    based upon the linear fit of the natural log of par with depth.

    Parameters
    ----------
    par: numpy.ndarray or pandas.Series
        The par data with units uE/m2/sec.
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    return_mask: bool
        If True, will return a mask for the photic layer
        (depth < euphotic depth).
    ref_percentage: int
        The percentage light depth to calculate the euphotic layer, typically
        assumed to be 1% of surface par.

    Returns
    -------
    light_depths: numpy.ndarray
        An array of the euphotic depths in metres.
    slopes: numpy.ndarray
        An array of the par attenuation coefficient (Kd).
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress

    def dive_slope(par, depth):
        mask = ~(np.isnan(par) | np.isnan(depth))
        x, y = depth[mask], par[mask]

        y = np.log(y)
        slope = linregress(x, y).slope

        return slope

    # Precentage light depth
    def dive_light_depth(depth, slope):

        if np.isnan(slope):
            euph_depth = np.nan
        else:
            light_depth = np.exp((depth * -1) / (-1 / slope)) * 100.0
            ind = abs(light_depth - ref_percentage).argmin()
            euph_depth = depth[ind]

        if return_mask:
            return depth < euph_depth
        else:
            return [euph_depth]

    #########################################################
    assert np.array(par).any(), "PAR does not contain data"

    slopes = []
    light_depths = []
    udives = np.unique(dives)
    for d in udives:
        i = dives == d
        zj = np.array(par[i])
        yj = np.array(depth[i])
        # xj = np.array(dives[i])

        if all(np.isnan(zj)):
            slope = np.nan
        else:
            slope = dive_slope(zj, yj)
        light_depth = dive_light_depth(yj, slope)

        slopes += (slope,)
        light_depths += (light_depth,)

    slopes = pd.Series(slopes, index=udives)
    light_depths = np.concatenate(light_depths)

    if not return_mask:
        light_depths = pd.Series(light_depths, index=udives)

    return light_depths, slopes

def colocate_MODIS_PAR(gliderfilename,pardata_directory):
    
    """
    Colocate MODIS PAR with glider positions
    
    Input directory to a gridded glider file with time, lats and lons
    and root directory of individual MODIS PAR .nc files
    
    """
    import xarray as xr
    import pandas as pd
    import numpy as np
    import os
    import math 

    df = xr.open_dataset(gliderfilename)
    # # Adjust latitudes to match the MODIS data
    df['Lon_adj'] = xr.where(df.longitude > 180, df.longitude - 360, df.longitude)
    # # Calculate julian date for float data
    a = np.asarray(df.time)
    dt = pd.to_datetime(a)
    glider_dates = dt.year * 1000 + dt.dayofyear
    
    
    # Satellite data list
    MODIS_dir = pardata_directory
    MODIS_list = sorted(os.listdir(MODIS_dir))
    MODIS_list = MODIS_list[:]
    sat_id = np.zeros(len(MODIS_list)) 
    sat_id[:] = np.nan
    for n in range(len(MODIS_list)):
    #    print(MODIS_list[n])
        a = MODIS_list[n].split('.')
        b = a[0].split('A')
        sat_id[n] = int(b[1])
    
    
    # Subsample MODIS data at nearest point to float profile using xarray capabilities
    par_mean = np.zeros(df.latitude.size) 
    par_mean[:] = np.nan
    lat_modis = np.zeros(df.latitude.size) 
    lat_modis[:] = np.nan
    lon_modis = np.zeros(df.latitude.size) 
    lon_modis[:] = np.nan
    date_modis=[]
    for n in range(df.latitude.size):
        # Select right satellite image to open
        diff = glider_dates[n] - sat_id
        idx = np.argwhere(diff==0)
        del diff
        if idx.size == 0 or math.isnan(df.latitude[n]) or math.isnan(df.Lon_adj[n]):
            par_mean[n] = -9999
            lat_modis[n] = -9999
            lon_modis[n] = -9999
            date_modis.append('NAT')
        else:
            satfilename = MODIS_dir + '/' + MODIS_list[int(idx)]
            ds = xr.open_dataset(satfilename)
            # Subselect satellite data at closed neighboring pixel to the point of interest        
            dssel = ds.sel(lat=df.latitude[n],lon=df.Lon_adj[n],method = "nearest")
            par_mean[n] = dssel.par
            lat_modis[n] = dssel.lat
            lon_modis[n] = dssel.lon
            date_modis.append(dssel.time_coverage_start)
    date_modis=pd.to_datetime(pd.Series(date_modis).squeeze())
    # Store in a dataset
    PAR_array = pd.Series(par_mean).squeeze()
    
    
    dpar=xr.Dataset(data_vars={'PAR':(('time'),PAR_array),
                               
                               },
                    coords={'time':(('time'),date_modis),
                        })

    dpar.PAR.attrs['units']='microEinstiens/m2/day'
    dpar.attrs['processing']='MODIS PAR colocated with glider lats and lons'

    return dpar
