import numpy as np
import matplotlib.pyplot as plt

bed_grl10_relaxed = np.load('grl10_bed_relaxed.npy')
surface_grl10 = np.load('grl10_surface.npy')
#bed_grl20_relaxed = np.load('grl20_bed_relaxed.npy')
#bed_grl40_relaxed = np.load('grl40_bed_relaxed.npy')

# mask ocean (all elevation values below 0)
mask = np.ones(np.shape(bed_grl10_relaxed), dtype=bool)
mask[bed_grl10_relaxed < 0] = False
bed_grl10_relaxed_masked = np.where(mask, bed_grl10_relaxed, 0)

# elevation plot
fig, axs = plt.subplots(1, 2, figsize=(8,4))
axs = axs.flatten()

im0 = axs[0].imshow(bed_grl10_relaxed_masked,
                    cmap='terrain',
                    origin='lower')
fig.colorbar(im0, label = "elevation [m]")
axs[0].set_title('bed')

im1 = axs[1].imshow(np.where(mask, surface_grl10, 0),
                    cmap='terrain',
                    origin='lower')
fig.colorbar(im1, label = "elevation [m]")
axs[1].set_title('surface')

fig.suptitle("Greenland Elevation Map")
fig.savefig('elevation.png', dpi=300, bbox_inches='tight')

#plt.figure(2)
#plt.imshow(bed_grl20_relaxed,cmap='bwr',origin='lower',vmin=-200,vmax=3000)
#plt.figure(3)
#plt.imshow(bed_grl40_relaxed,cmap='bwr',origin='lower',vmin=-200,vmax=3000)
