# %%
#####################################
# Positive Degree Day Glacier Model #
# Robert Wright                     #
# 01/24                             #
#####################################

# python libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set_theme(style='ticks', color_codes=True)
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
# model parameters
import params as p

# read orography & lat-lon bounds
# "switch" dimensions to x = lon & y = lat
grl  = np.transpose(np.load('../model-input/grl20_surface.npy'))
xlon = np.transpose(np.load('../model-input/grl20_lon.npy'))
ylat = np.transpose(np.load('../model-input/grl20_lat.npy'))
#plt.imshow(grl20,cmap='bwr',origin='lower',vmin=-200,vmax=3000)
#plt.colorbar()

trans = ccrs.PlateCarree()
proj = ccrs.Stereographic(central_latitude=90, central_longitude=316)

# add coastline of greenland
# get path to shapefile
shpfilename = shpreader.natural_earth(resolution='50m',
                                      category='cultural',
                                      name='admin_0_countries')
# read shapefile
shpfile = shpreader.Reader(shpfilename)
# select greenland only
greenland = [country.geometry for country in shpfile.records() if country.attributes["NAME_LONG"] == "Greenland"]

# mask ocean (all elevation values below 0)
seamask = np.ones(np.shape(grl), dtype=bool)
seamask[grl > 0] = False

### ELEVATION ###
# set minimum elevation to 0 (no negative values)
elev = np.where(grl>0, grl, 0)
# get dimensions of elevation file & add time axis in days
dim = np.shape(elev) + (365,)

### MODEL SETUP ###
print("MODEL PARAMETERS")
print("> model dimension:", dim, "(lon, lat, day)")
print("Temperature [°C]")
print("> weather scaling =", p.T_WEATHER_SCALE)
print("> lapse rate =", p.LAPSE_RATE)
print("> latitudinal difference =", p.T_LATITUDE_DIFF)
print("> seasonal difference =", p.T_SEASONAL_DIFF)
print("Precipitation [mm/day]")
print("> latitudinal difference =", p.PR_LATITUDE_DIFF, '\n')


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
t = np.full(dim, p.T_INITIAL)

def weather(x, scale=p.T_WEATHER_SCALE):
    """
    add randomness (weather) to temperature by drawing
    values from normal distribution
    """
    # (scale = standard deviation)
    w = np.random.normal(loc=0, scale=scale, size=dim)
    x = x + w
    return x

def elevation(x, lapse_rate=p.LAPSE_RATE):
    """
    scale input temperature field by elevation
    """
    # loop over all days
    for tday in range(dim[2]):
        x[:,:,tday] = x[:,:,tday] - (lapse_rate * elev/100)
    return x

def latitude(x, diff=p.T_LATITUDE_DIFF):
    """
    scale input temperature field by latitude
    """
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for yylat in range(dim[1]):
            x[:,yylat,tday] = x[:,yylat,tday] - (yylat/dim[1] * diff)
    return x

def season(x, diff=p.T_SEASONAL_DIFF):
    """
    add seasonal cycle to temperature
    """
    # use cosine to model seasonal cycle
    dd = np.linspace(0, 2*np.pi, 365)
    for xxlon in range(dim[0]):
        for yylat in range(dim[1]):
            x[xxlon, yylat, :] = x[xxlon, yylat, :] + (-1) * np.cos(dd) * (diff/2)
    return x

# apply all scalings to original temperature field t
tfinal = latitude(elevation(season(weather((t)))))

# CONTROL TEMPERATURE FIELD
print("TEMPERATURE FIELD")
print("> mean annual temperature:",np.round(np.mean(tfinal),2),'°C\n')

# compute temporal mean (i.e., throughout full year)
tfinal_tmean = np.mean(tfinal, axis=2)

def plot_greenland(field, title, figpath,
                   cmesh_kw=dict(cmap='viridis'),
                   cbar_kw=dict()):
    """
    general routine to plot greenland maps
    """
    # sea-mask input field
    plotme = np.ma.masked_where(seamask, field)
    # plotting...
    fig, ax = plt.subplots(subplot_kw=dict(projection = proj))
    cm = ax.pcolormesh(xlon, ylat, plotme, shading='nearest',
                       transform=trans, **cmesh_kw)
    cb = fig.colorbar(cm, **cbar_kw)
    ax.set_title(title)
    ax.add_geometries(greenland, crs=trans, fc='none',
                      ec='k', alpha=0.8)
    ax.gridlines(alpha=.6, ls='--', lw=.5)
    fig.savefig(figpath, dpi=300, bbox_inches="tight")
    plt.show()

# plot temperature field
plot_greenland(tfinal_tmean,
               title="PDD: Input temperature field",
               cbar_kw=dict(label="t [°C]"),
               cmesh_kw=dict(cmap='jet'),
               figpath="figs/pdd-temperature-field.png")


### CREATE PRECIPITATION FIELD pr with ###
# a) latitudinal gradient
#    south-north absolute difference: 4 mm d-1 (era5)
# b) mean annual precipitation: pr_mean = 1.6 mm d-1 (era5)
#
# The scaling parameters are slightly adjusted in the following...

# create precipitation field analogous to temperature field

pr = np.full(dim, p.PR_INITIAL)
prfinal = latitude(pr, diff=p.PR_LATITUDE_DIFF)

# CONTROL PRECIPITATION FIELD
print("PRECIPITATION FIELD")
print("> mean annual precipitation:",np.mean(prfinal),'mm d-1\n')

# compute temporal mean (i.e., throughout full year)
prfinal_tmean = np.mean(prfinal, axis=2)

plot_greenland(prfinal_tmean,
               title="PDD: Input precipitation field",
               cmesh_kw=dict(cmap="Blues"),
               cbar_kw=dict(label=r"pr [mm d$^{-1}$]"),
               figpath="figs/pdd-precipitation-field.png")


### SURFACE MASS BALANCE ###
# 1. Compute ablation ABL by multiplying PDD
#    & melting factor beta
# 2. Compute accumulation ACC by summing up precipitation
#    at temperatures below 0°C
# 3. SMB = ACC - ABL 

# no time domain, hence:
smb_dim = np.shape(elev)

# PDD / ABLATION
pdd = np.zeros(smb_dim)
for xxlon in range(dim[0]):
    for yylat in range(dim[1]):
        # sum up positive temperature values per grid cell
        pdd[xxlon,yylat] = np.sum(a = tfinal[xxlon,yylat,:],
                                  where = tfinal[xxlon,yylat,:]>0)
abl = p.BETA * pdd

# control plot
#fig, ax = plt.subplots()
#im = ax.imshow(np.transpose(pdd), origin='lower')
#cb = fig.colorbar(im, label='PDD [°C]')

# ACCUMULATION
acc = np.zeros(smb_dim)
for xxlon in range(dim[0]):
    for yylat in range(dim[1]):
        # sum up precipitation (if t<0) per grid cell
        acc[xxlon,yylat] = np.sum(a = prfinal[xxlon,yylat,:],
                                  where = tfinal[xxlon,yylat,:]<0)
        
# control plot
#fig, ax = plt.subplots()
#im = ax.imshow(np.transpose(acc), origin='lower')
#cb = fig.colorbar(im, label='ACC [mm d-1]')

smb = acc - abl
# FIXME: I guess the units are [mm yr-1] now, but why?

print("SURFACE MASS BALANCE")
print("> pdd factor =", p.BETA)
print("> mean surface mass balance =",np.mean(smb),"[mm yr-1]")

plot_greenland(field=smb,
               title="PDD: Surface mass balance",
               figpath="figs/pdd-smb.png",
               cmesh_kw=dict(cmap='RdBu',
                             norm=colors.TwoSlopeNorm(vcenter=0,vmin=-5000)),
               cbar_kw=dict(extend='min',
                            label=r"smb [mm yr$^{-1}$]"))

# %%
