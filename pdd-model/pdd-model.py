# Positive Degree Day Glacier Model
# Robert Wright
# 01/24

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

grl20 = np.load('../model_input/grl10_surface.npy')
#plt.imshow(grl20,cmap='bwr',origin='lower',vmin=-200,vmax=3000)
#plt.colorbar()

### ELEVATION ###
# set minimum elevation to 0 (no negative values)
elev = np.where(grl20>0, grl20, 0)
# get dimensions of elevation file & add time axis in days
dim = np.shape(elev) + (365,)
print("> model dimension:", dim, "(lat, lon, day)\n")

### CREATE TEMPERATURE FIELD t with ###
# a) latitudinal gradient
#    south-north absolute difference: 15K (era5)
# b) decrease with elevation
#    elevation lapse rate: 0.7K/100m (literature)
# c) a seasonal cycle 
#    winter-summer difference: 20K (era5)
# d) some randomness to represent weather
#    normal distribution, sigma = 3.5K (era5)
# e) mean annual temperature: T_mean = -10.5°C (era5)

# for now, choose filling value to reach appropriate
# mean annual temperature
t = np.full(dim, 4)

def weather(x):
    """
    add randomness (weather) to temperature by drawing
    values from normal distribution
    """
    # (scale = standard deviation)
    w = np.random.normal(loc=0, scale=3.5, size=dim)
    x = x + w
    return x

def elevation(x):
    """
    scale input temperature field by elevation
    (elevation lapse rate: 0.7°C / 100m)
    """
    for tday in range(dim[2]):
        x[:,:,tday] = x[:,:,tday] - (0.7 * elev/100)
    return x

def latitude(x):
    """
    scale input temperature field by latitude
    (south-north absolute difference: 15°C)
    """
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for xlat in range(dim[0]):
            x[xlat,:,tday] = x[xlat,:,tday] - (xlat/dim[0] * 15)
    return x

def season(x):
    """
    add seasonal cycle to temperature
    (summer-winter difference of 20°C)
    """
    diff = 20
    # use cosine to model seasonal cycle
    dd = np.linspace(0, 2*np.pi, 365)
    for xlat in range(dim[0]):
        for ylon in range(dim[1]):
            x[xlat, ylon, :] = x[xlat, ylon, :] + (-1) * np.cos(dd) * (diff/2)
    return x

# apply all scalings to original temperature field t
tfinal = latitude(elevation(season(weather((t)))))

# CONTROL TEMPERATURE FIELD
print("> mean annual temperature:",np.mean(tfinal),'°C\n')

# plot temperature field
fig, ax = plt.subplots()
# compute temporal mean (i.e., throughout full year)
tfinal_tmean = np.mean(tfinal, axis=2)
im = ax.imshow(tfinal_tmean, origin="lower", cmap="viridis")
cb = fig.colorbar(im, label='t [°C]')
ax.set_title("PDD: input temperature field")
fig.savefig("figs/pdd-temperature-field.png", dpi=300, bbox_inches="tight")
plt.show()


### CREATE TEMPERATURE FIELD pr with ###
# a) latitudinal gradient
#    south-north absolute difference: 4 mm d-1 (era5)
# b) mean annual precipitation: pr_mean = 1.6 mm d-1 (era5)
#
# The scaling parameters are slightly adjusted in the following:

# create precipitation field analogous to temperature field
pr = np.full(dim, 3.5)

def latitude_pr(x, diff=3.5):
    """
    scale input precipitation field by latitude
    diff: south-north absolute difference
    """
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for xlat in range(dim[0]):
            x[xlat,:,tday] = x[xlat,:,tday] - (xlat/dim[0] * diff)
    return x

prfinal = latitude_pr(pr)

# CONTROL PRECIPITATION FIELD
print("> mean annual precipitation:",np.mean(prfinal),'mm d-1\n')

# plot precipitation field
fig, ax = plt.subplots()
# compute temporal mean (i.e., throughout full year)
prfinal_tmean = np.mean(prfinal, axis=2)
im = ax.imshow(prfinal_tmean, origin="lower", cmap="viridis")
cb = fig.colorbar(im, label=r'pr [mm d$^{-1}$]')
ax.set_title("PDD: input precipitation field")
fig.savefig("figs/pdd-precipitation-field.png", dpi=300, bbox_inches="tight")
plt.show()

### SURFACE MASS BALANCE ###
#
# 1. Compute ablation ABL by multiplying PDD
#    & melting factor beta
# 2. Compute accumulation ACC by summing up precipitation
#    at temperatures below 0°C
# 3. SMB = ACC - ABL 

# no time domain, hence:
smb_dim = np.shape(elev)

# MELTING / ABLATION
BETA = 8 # mm d-1 K-1

pdd = np.zeros(smb_dim)
for xlat in range(dim[0]):
    for ylon in range(dim[1]):
        # sum up positive temperature values per grid cell
        pdd[xlat,ylon] = np.sum(tfinal[xlat,ylon,:],
                                where=tfinal[xlat,ylon,:]>0)
abl = BETA * pdd

# control plot
#fig, ax = plt.subplots()
#im = ax.imshow(pdd, origin='lower')
#cb = fig.colorbar(im, label='PDD [°C]')

# ACCUMULATION
acc = np.zeros(smb_dim)
for xlat in range(dim[0]):
    for ylon in range(dim[1]):
        # sum up precipitation (if t<0) per grid cell
        acc[xlat,ylon] = np.sum(prfinal[xlat,ylon,:],
                                where=tfinal[xlat,ylon,:]<0)
        
# control plot
#fig, ax = plt.subplots()
#im = ax.imshow(acc, origin='lower')
#cb = fig.colorbar(im, label='ACC [mm d-1]')

smb = acc - abl
# I think the units are [mm yr-1] now (?)

fig, ax = plt.subplots()
norm = colors.CenteredNorm(halfrange=1200)
im = ax.imshow(smb, origin='lower', norm=norm, cmap="bwr_r")
cb = fig.colorbar(im, label=r"smb [mm yr$^{-1}$]", extend="min")
ax.set_title("Surface mass balance (PDD model)")
fig.savefig("figs/pdd-smb.png", dpi=300, bbox_inches="tight")