import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

bed_grl10_relaxed = np.load('grl10_bed_relaxed.npy')
bed_grl10 = np.load('grl10_bed.npy')
surface_grl10 = np.load('grl10_surface.npy')
#bed_grl20_relaxed = np.load('grl20_bed_relaxed.npy')
#bed_grl40_relaxed = np.load('grl40_bed_relaxed.npy')

# mask ocean (all elevation values below 0)
mask = np.zeros(np.shape(bed_grl10_relaxed), dtype=bool)
mask[bed_grl10_relaxed < 0] = True
bed_grl10_relaxed = np.ma.masked_array(bed_grl10_relaxed, mask=mask)
bed_grl10 = np.ma.masked_array(bed_grl10, mask=mask)
surface_grl10 = np.ma.masked_array(surface_grl10, mask=mask)

# elevation plot
fig, axs = plt.subplots(1, 3, figsize=(8,5))
axs = axs.flatten()

topography = [bed_grl10_relaxed, bed_grl10, surface_grl10]
titles = ['bedrock relaxed', 'bedrock under ice sheet', 'ice sheet surface']

for i, ax in enumerate(axs):
    im = ax.imshow(topography[i], cmap='terrain', origin='lower',
                   norm=mpl.colors.Normalize(vmin=0, vmax=3000))
    ax.set_title(titles[i])

fig.colorbar(im, ax=axs, orientation='horizontal', label='elevation [m]', extend='both')
fig.suptitle("Greenland Elevation Map")
fig.savefig('elevation.png', dpi=300, bbox_inches='tight')
