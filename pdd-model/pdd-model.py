# Positive Degree Day Glacier Model
# Robert Wright
# 01/24

import numpy as np
import matplotlib.pyplot as plt

grl20 = np.load('../model_input/grl10_surface.npy')
#plt.imshow(grl20,cmap='bwr',origin='lower',vmin=-200,vmax=3000)
#plt.colorbar()

# ELEVATION
# set minimum elevation to 0 (no negative values)
elev = np.where(grl20>0, grl20, 0)
# get dimensions of elevation file & # add time axis in days
dim = np.shape(elev) + (365,)
print("model dimension:", dim, "(lat, lon, day)")

# CREATE TEMPERATURE FIELD t with
# a) latitudinal gradient
#    south-north absolute difference: 10K
# b) decrease with elevation
#    elevation lapse rate: 0.7K/100m
# c) a seasonal cycle
# d) some randomness to represent weather

t = np.full(dim, 5)

def weather(x):
    """
    add randomness (weather) to temperature by drawing
    values from normal distribution
    """
    # (scale = standard deviation)
    w = np.random.normal(loc=0, scale=2, size=dim)
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
    (south-north absolute difference: 10°C)
    """
    # loop over days
    for tday in range(dim[2]):
        # loop over latitudes
        for xlat in range(dim[0]):
            x[xlat,:,tday] = x[xlat,:,tday] - (xlat/dim[0] * 10)
    return x

def season(x):
    """
    add seasonal cycle to temperature
    (summer-winter difference of 15°C)
    """
    diff = 15
    # use cosine to model seasonal cycle
    dd = np.linspace(0, 2*np.pi, 365)
    for xlat in range(dim[0]):
        for ylon in range(dim[1]):
            x[xlat, ylon, :] = x[xlat, ylon, :] + (-1) * np.cos(dd) * (diff/2)
    return x

#tt = latitude(elevation(season(weather((t)))))

#tfinal = random(season(elevation(latitude(t))))

#fig, ax = plt.subplots()
#im = ax.imshow(tt[:,:,182], origin="lower", cmap="viridis")
#cb = fig.colorbar(im)
#fig.savefig("temperature-field.png", dpi=300, bbox_inches="tight")
#plt.show()


# CREATE TEMPERATURE FIELD pr with