################################
# Compute Surface Mass Balance #
# Robert Wright                #
# 05/24                        #
################################

# python libraries
import sys
import numpy as np

def output_smb_params(parpath):
    """
    print model parameters to console
    parameters:
    parpath: relative path to parameter file
    """
    # there should be a better way to do this
    sys.path.append(parpath)
    import params as p

    print("SURFACE MASS BALANCE:")
    print("> pdd factor =", p.BETA)
    print("Temperature [°C]")
    print("> initial value =", p.T_INITIAL)
    print("> weather scaling =", p.T_WEATHER_SCALE)
    print("> lapse rate =", p.LAPSE_RATE)
    print("> latitudinal difference =", p.T_LATITUDE_DIFF)
    print("> seasonal difference =", p.T_SEASONAL_DIFF)
    print("Precipitation [mm/day]")
    print("> initial value =", p.PR_INITIAL)
    print("> latitudinal difference =", p.PR_LATITUDE_DIFF)
    print("> distance weight =", p.PR_COASTLINE, '\n')

def output_smb_params_to_logger(parpath, logger):
    """
    print model parameters to logger
    parameters:
    parpath: relative path to parameter file
    logger: logging object
    """
    # there should be a better way to do this
    sys.path.append(parpath)
    import params as p

    logger.info("SURFACE MASS BALANCE:")
    logger.info("> pdd factor = %s", p.BETA)
    logger.info("Temperature [°C]")
    logger.info("> initial value = %s", p.T_INITIAL)
    logger.info("> weather scaling = %s", p.T_WEATHER_SCALE)
    logger.info("> lapse rate = %s", p.LAPSE_RATE)
    logger.info("> latitudinal difference = %s", p.T_LATITUDE_DIFF)
    logger.info("> seasonal difference = %s", p.T_SEASONAL_DIFF)
    logger.info("Precipitation [mm/day]")
    logger.info("> initial value = %s", p.PR_INITIAL)
    logger.info("> latitudinal difference = %s", p.PR_LATITUDE_DIFF)
    logger.info("> distance weight = %s\n", p.PR_COASTLINE)

def compute_smb(z, seamask, parpath, distance, verbose=False):
    """
    compute surface mass balance in [m yr-1]
    parameters:
    z: surface elevation field [m] (2D array)
    seamask: boolean array to mask ocean (2D array)
    """
    sys.path.append(parpath)
    import params as p

    # get dimensions of elevation file & add time axis in days
    dim = np.shape(z) + (365,)

    ### CREATE TEMPERATURE FIELD t with ###
    # a) latitudinal gradient
    #    south-north absolute difference: 15°C (era5)
    # b) decrease with elevation
    #    elevation lapse rate: 0.7K/100m (literature)
    # c) a seasonal cycle
    #    winter-summer difference: 20K (era5)
    # d) some randomness to represent weather
    #    normal distribution, sigma = 3.5K (era5)
    # e) mean annual temperature: T_mean = -10.5°C (era5)

    # for now, choose filling value to reach appropriate
    # mean annual temperature
    t = np.full(dim, p.T_INITIAL, dtype=float)

    # add randomness (weather) to temperature by drawing
    # values from normal distribution
    #np.random.seed(42)
    # (scale = standard deviation)
    w = np.random.normal(loc=0, scale=p.T_WEATHER_SCALE, size=dim)
    t = t + w

    # scale input temperature field by elevation
    # loop over all days
    for tday in range(dim[2]):
        t[:,:,tday] = t[:,:,tday] - (p.LAPSE_RATE * z/100)

    # scale input temperature field by latitude
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for jlat in range(dim[1]):
            t[:,jlat,tday] = t[:,jlat,tday] - (p.T_LATITUDE_DIFF * (jlat+1)/dim[1])


    # add seasonal cycle to temperature field
    # use cosine to model seasonal cycle
    dd = np.linspace(0, 2*np.pi, 365)
    for xxlon in range(dim[0]):
        for yylat in range(dim[1]):
            t[xxlon, yylat, :] = t[xxlon, yylat, :] + \
                                 (-1) * np.cos(dd) * (p.T_SEASONAL_DIFF/2)

    # CONTROL TEMPERATURE FIELD
    if verbose:
        print("TEMPERATURE FIELD")
        print("> mean annual temperature:",np.round(np.mean(t),2),'°C\n')

    ### CREATE PRECIPITATION FIELD pr with ###
    # a) latitudinal gradient
    #    south-north absolute difference: 4 mm d-1 (era5)
    # b) mean annual precipitation: pr_mean = 1.6 mm d-1 (era5)
    # c) distance from coastline
    #
    # (scaling parameters are slightly adjusted in the following...)

    # create precipitation field analogous to temperature field
    pr = np.full(dim, p.PR_INITIAL, dtype=float)

    # scale input precipitation field by latitude
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for jlat in range(dim[1]):
            pr[:,jlat,tday] = pr[:,jlat,tday] - (p.PR_LATITUDE_DIFF * (jlat+1)/dim[1])

    # loop over all days
    for tday in range(dim[2]):
        pr[:,:,tday] = pr[:,:,tday] / (p.PR_COASTLINE * distance + 1)

    # CONTROL PRECIPITATION FIELD
    if verbose:
        print("PRECIPITATION FIELD")
        print("> mean annual precipitation:",np.mean(pr),'mm d-1\n')

    ### SURFACE MASS BALANCE ###
    # 1. Compute ablation ABL by multiplying PDD
    #    & melting factor beta
    # 2. Compute accumulation ACC by summing up precipitation
    #    at temperatures below 0°C
    # 3. SMB = ACC - ABL

    # no time domain, hence:
    smb_dim = np.shape(z)

    # PDD / ABLATION
    pdd = np.zeros(smb_dim)
    for xxlon in range(dim[0]):
        for yylat in range(dim[1]):
            # sum up positive temperature values per grid cell
            pdd[xxlon,yylat] = np.sum(a = t[xxlon,yylat,:],
                                      where = t[xxlon,yylat,:]>0)
    abl = p.BETA * pdd

    # ACCUMULATION
    acc = np.zeros(smb_dim)
    for xxlon in range(dim[0]):
        for yylat in range(dim[1]):
            # sum up precipitation (if t<0) per grid cell
            acc[xxlon,yylat] = np.sum(a = pr[xxlon,yylat,:],
                                      where = t[xxlon,yylat,:]<0)

    # surface mass balance by subtracting ablation from accumulation
    smb = (acc - abl) / 1e3
    # I guess the units are [m yr-1] now, but why?

    # apply bounds for smb
    smb[smb > 2] = 2
    smb[smb < -2] = -2

    # CONTROL SURFACE MASS BALANCE
    if verbose:
        print("SURFACE MASS BALANCE")
        print("> pdd factor =", p.BETA)
        print("> mean surface mass balance =",
              # masking ocean (!)
              np.mean(np.ma.masked_where(seamask, smb)),
              "[m yr-1]")

    return smb

def compute_smb_NorESM(ERA5_t, ERA5_pr, NorESM_t, NorESM_pr, clim_index, year, beta=5):
    """
    compute surface mass balance on climate forcing derived from Norwegian Earth System Model
    paramters:
    ERA5/NorESM: fields (lon, lat, time); temperature [°C] and precipitation [mm d-1]
    clim_index: climate index for future years (1D array)
    beta: pdd factor (default: 5)
    """
    # The climate forcing for a given year is the sum of the present-day climatology and the climate anomaly of the year 2100,
    # scaled to match the current year.

    # You can define the anomalous climate of a given year using the climate index:
    t2m_anom    = NorESM_t  * clim_index[year]
    precip_anom = NorESM_pr * clim_index[year]

    # Since this is only an anomaly with regard to the present day, the full (absolute) climate forcing for a given year is:
    t2m_now     = ERA5_t  + t2m_anom
    precip_now  = ERA5_pr + precip_anom

    ### SURFACE MASS BALANCE ###
    # 1. Compute ablation ABL by multiplying PDD
    #    & melting factor beta
    # 2. Compute accumulation ACC by summing up precipitation [mm d-1]
    #    at temperatures below 0°C
    # 3. SMB = ACC - ABL

    # no time domain, hence:
    smb_dim = np.shape(ERA5_t)[:-1]

    # PDD / ABLATION
    pdd = np.zeros(smb_dim)
    for ilon in range(smb_dim[0]):
        for jlat in range(smb_dim[1]):
            # sum up positive temperature values per grid cell
            pdd[ilon,jlat] = np.sum(a = t2m_now[ilon,jlat,:],
                                    where = t2m_now[ilon,jlat,:]>0)
    abl = beta * pdd

    # ACCUMULATION
    acc = np.zeros(smb_dim)
    for ilon in range(smb_dim[0]):
        for jlat in range(smb_dim[1]):
            # sum up precipitation (if t<0) per grid cell
            acc[ilon,jlat] = np.sum(a = precip_now[ilon,jlat,:],
                                    where = t2m_now[ilon,jlat,:]<0)

    # surface mass balance by subtracting ablation from accumulation
    smb = (acc - abl) / 1e3
    # I guess the units are [m yr-1] now, but why?

    # apply bounds for smb
    # actually, this is not necessary, since temperature and precipitation fields are realistic
    #smb[smb > 2] = 2
    #smb[smb < -2] = -2

    return smb

def compute_distance_coastline(z):
    """
    compute (euclidian) distance from coastline
    parameters:
    z: surface elevation field [m] (2D array)
    """
    # transform coastline into boolean array, where 1 represents coastline
    # height limits are arbitrary and related to chosen resolution
    coastline = np.where((z<250) & (z>-50), 1, 0)
    distance = np.zeros_like(coastline)
    for ilon in range(z.shape[0]):
        for jlat in range(z.shape[1]):
            # can't compute distance to coastline for points on coastline
            if coastline[ilon,jlat] == 0:
                # use euclidian distance as metric
                distance[ilon,jlat] = np.min([np.sqrt((ilon - x)**2 + (jlat - y)**2) \
                                              for x in range(z.shape[0]) for y in range(z.shape[1]) \
                                              if coastline[x, y] == 1])
    np.save('distance-coastline', distance)

#####################################################################################################

# somehow, the following functions didn't speed things up, in fact, they didn't work at all...

def temperature_field(z, parpath, verbose):
    """
    compute temperature field
    parameters:
    z: surface elevation field [m] (2D array)
    """
    # read parameters
    sys.path.append(parpath)
    import params as p

    # get dimensions of elevation file & add time axis in days
    dim = np.shape(z) + (365,)

    ### CREATE TEMPERATURE FIELD t with ###
    # a) latitudinal gradient
    #    south-north absolute difference: 15°C (era5)
    # b) decrease with elevation
    #    elevation lapse rate: 0.7K/100m (literature)
    # c) a seasonal cycle
    #    winter-summer difference: 20K (era5)
    # d) some randomness to represent weather
    #    normal distribution, sigma = 3.5K (era5)
    # e) mean annual temperature: T_mean = -10.5°C (era5)

    # for now, choose filling value to reach appropriate
    # mean annual temperature
    t = np.full(dim, p.T_INITIAL, dtype=float)

    # add randomness (weather) to temperature by drawing
    # values from normal distribution
    # (scale = standard deviation)
    w = np.random.normal(loc=0, scale=p.T_WEATHER_SCALE, size=dim)
    t = t + w

    # use cosine to model seasonal cycle
    dd = np.linspace(0, 2*np.pi, 365)
    
    # loop over all days
    for tday in range(dim[2]):
        # scale input temperature field by elevation
        t[:,:,tday] = t[:,:,tday] - (p.LAPSE_RATE * z/100)
        # loop over latitudes
        for jlat in range(dim[1]):
            # scale input temperature field by latitude
            t[:,jlat,tday] = t[:,jlat,tday] - (p.T_LATITUDE_DIFF * (jlat+1)/dim[1])
            # loop over longitudes
            for ilon in range(dim[0]):
                # add seasonal cycle to temperature field
                t[ilon, jlat, :] = t[ilon, jlat, :] + \
                                   (-1) * np.cos(dd) * (p.T_SEASONAL_DIFF/2)
    # CONTROL TEMPERATURE FIELD
    if verbose:
        print("TEMPERATURE FIELD")
        print("> mean annual temperature:",np.round(np.mean(t),2),'°C\n')

    return t

def precipitation_field(z, parpath, verbose):
    """
    compute precipitation field
    parameters:
    z: surface elevation field [m] (2D array)
    """
    # read parameters
    sys.path.append(parpath)
    import params as p

    # get dimensions of elevation file & add time axis in days
    dim = np.shape(z) + (365,)

    ### CREATE PRECIPITATION FIELD pr with ###
    # a) latitudinal gradient
    #    south-north absolute difference: 4 mm d-1 (era5)
    # b) mean annual precipitation: pr_mean = 1.6 mm d-1 (era5)
    # c) distance from coastline

    # create precipitation field analogous to temperature field
    pr = np.full(dim, p.PR_INITIAL, dtype=float)
    # import distance to coastline
    distance = np.load('distance-coastline.npy')

    # loop over days
    for tday in range(dim[2]):
        # scale by distance from coastline
        pr[:,:,tday] = pr[:,:,tday] / (p.PR_COASTLINE * distance + 1)
        # loop over latitudes
        for jlat in range(dim[1]):
            # scale input precipitation field by latitude
            pr[:,jlat,tday] = pr[:,jlat,tday] - (p.PR_LATITUDE_DIFF * (jlat+1)/dim[1])

    # CONTROL PRECIPITATION FIELD
    if verbose:
        print("PRECIPITATION FIELD")
        print("> mean annual precipitation:",np.mean(pr),'mm d-1\n')

    return pr

def compute_smb_from_fields(z, t, pr, seamask, parpath, verbose):
    """
    compute surface mass balance in [m yr-1]
    parameters:
    z: surface elevation field [m] (2D array)
    t: temperature field [°C] (3D array)
    pr: precipitation field [mm d-1] (3D array)
    """
    # read parameters
    sys.path.append(parpath)
    import params as p

    ### SURFACE MASS BALANCE ###
    # 1. Compute ablation ABL by multiplying PDD
    #    & melting factor beta
    # 2. Compute accumulation ACC by summing up precipitation
    #    at temperatures below 0°C
    # 3. SMB = ACC - ABL

    # no time domain, hence:
    smb_dim = np.shape(z)

    # PDD / ABLATION
    pdd = np.zeros(smb_dim)
    for ilon in range(smb_dim[0]):
        for jlat in range(smb_dim[1]):
            # sum up positive temperature values per grid cell
            pdd[ilon,jlat] = np.sum(a = t[ilon,jlat,:],
                                    where = t[ilon,jlat,:]>0)
    abl = p.BETA * pdd

    # ACCUMULATION
    acc = np.zeros(smb_dim)
    for ilon in range(smb_dim[0]):
        for jlat in range(smb_dim[1]):
            # sum up precipitation (if t<0) per grid cell
            acc[ilon,jlat] = np.sum(a = pr[ilon,jlat,:],
                                    where = t[ilon,jlat,:]<0)

    # surface mass balance by subtracting ablation from accumulation
    smb = (acc - abl) / 1e3
    # I guess the units are [m yr-1] now, but why?

    # apply bounds for smb
    smb[smb > 2] = 2
    smb[smb < -2] = -2

    # CONTROL SURFACE MASS BALANCE
    if verbose:
        print("SURFACE MASS BALANCE")
        print("> pdd factor =", p.BETA)
        print("> mean surface mass balance =",
              # masking ocean (!)
              np.mean(np.ma.masked_where(seamask, smb)),
              "[m yr-1]")

    return smb

#####################################################################################################
