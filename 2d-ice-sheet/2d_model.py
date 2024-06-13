################################
# 2D Greenland Ice Sheet Model #
# Robert Wright                #
# 03/2024                      #
################################

# ToDO:
# - berechne die Geschwindigkeit u des Eises
# - wie interaktiven plot als animation speichern?
# - case synthax instead of smb if/else flags

from datetime import timedelta, datetime
import logging
import sys
import cProfile
import pstats
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from surface_mass_balance import output_smb_params_to_logger, compute_smb, compute_smb_NorESM
sns.set_theme(style='ticks', color_codes=True)
np.random.seed(42)

# simulation configuration
#SIM_PATH = 'climate-anomalies/deltaT_deltaP/+12_8'
SIM_PATH = sys.argv[1]
SAVE_RESULTS = True # save model results to npz file
# set up smb (default: idealized smb)
BENCHMARK = False # used for benchmark simulation with flat bedrock and
                  # constant surface mass balance
REALISTIC_SMB = False # use realistic surface mass balance data computed from NorESM

### SET UP MODEL ###############################################################
def import_data():
    # read orography & lat-lon bounds
    # "switch" dimensions to x = lon & y = lat
    bedrock  = np.transpose(np.load('../model-input/grl40_bed.npy'))
    surface  = np.transpose(np.load('../model-input/grl40_surface.npy'))
    # create mask for ocean before flat bedrock option is called
    # (derive mask from relaxed bedrock, otherwise, negative elevation values inside of Greenland)
    relaxed_bed = np.transpose(np.load('../model-input/grl40_bed_relaxed.npy'))
    seamask = np.zeros(np.shape(relaxed_bed), dtype=bool)
    # (all elevation values below 0)
    seamask[relaxed_bed < 0] = True
    # set bedrock to zero for flat bedrock simulation
    if BENCHMARK:
        bedrock = np.zeros_like(bedrock)
        surface = np.zeros_like(surface)
    # read lat-lon bounds (error in original lon file, use csv instead of npy)
    xlon = np.loadtxt('../model-input/grl40_lon.csv') # no transpose needed
    ylat = np.transpose(np.load('../model-input/grl40_lat.npy'))
    return bedrock, surface, seamask, xlon, ylat

def set_model_parameters(bedrock):
    # grid spacing and time step
    nx = bedrock.shape[0] # number of grid points in x-direction
    dx = 40*1e3           # grid spacing [m]
    ny = bedrock.shape[1] # number of grid points in y-direction
    dy = dx           # grid spacing [m]
    nt = 10000 # number of time steps
    dt = 2     # time step [yr]
    if REALISTIC_SMB:
        dt = 1    # climate index is provided for each year
        nt = 1000 # run model for 1000 years
    # simulation parameters
    A   = 1e-24    # flow parameter [s^-1 Pa^-3]
    RHO = .92 *1e3 # ice density [kg m^-3]
    G   = 9.81     # gravitational acceleration [m s^-2]
    return nx, ny, dx, dy, nt, dt, A, RHO, G

def output_model_parameters_to_logger(nx, ny, dx, dy, dt, nt, logger):
    logger.info("MODEL PARAMETERS:")
    logger.info("> dimensions = {}, {} (x, y)".format(nx, ny))
    logger.info("> dx, dy = {}, {} km".format(dx/1e3, dy/1e3))
    logger.info("> dt = {} yr".format(dt))
    logger.info("> nt = {}\n".format(nt))

def logging_setup():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a file handler (open in write mode to overwrite previous log file)
    file_handler = logging.FileHandler(f'{SIM_PATH}/console_output.log', mode='w')
    file_handler.setLevel(logging.INFO)
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

### INITIALIZE MODEL ###########################################################
def initialize_model(nt, nx, ny, bedrock, surface):
    # initializations of model variables
    z = np.tile(surface, (nt, 1, 1)) # elevation [m]
    h = np.tile(surface-bedrock, (nt, 1, 1)) # ice thickness [m]
    smb = np.zeros((nt, nx, ny)) # surface mass balance [m/yr]
    u_x = np.zeros((nt, nx, ny)) # ice velocity [m/yr]
    u_y = np.zeros((nt, nx, ny))
    u_abs = np.zeros((nt, nx, ny)) # absolute ice velocity [m/yr]
    Fmx, Fpx, Fmy, Fpy = np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny)) # ice fluxes [m^2/yr]
    return z, h, smb, u_x, u_y, u_abs, Fmx, Fpx, Fmy, Fpy

### SURFACE MASS BALANCE #######################################################    
def smb_benchmark(bedrock):
    # find the center of the bedrock array
    center_row = bedrock.shape[0] // 2
    center_col = bedrock.shape[1] // 2
    # initialize SMB with -2 m/yr everywhere
    smb = np.full_like(bedrock, -2)
    # set SMB to 2 m/yr in the 10x10 grid box square at the center
    smb[center_row-5:center_row+5, center_col-5:center_col+5] = 2
    return smb

def load_distance_to_coast():
    # import distance to coastline
    distance = np.load('distance_coastline_grl40.npy')
    return distance

def load_climate_data():
    # load climate data for realistic smb
    # swap dimensions to x = lon & y = lat
    ERA5_t2m    = np.transpose(np.load('../model-input/grl40_ERA5_t2m.npy'), axes=(1,0,2))
    ERA5_prec   = np.transpose(np.load('../model-input/grl40_ERA5_prec.npy'), axes=(1,0,2))
    NorESM_t2m  = np.transpose(np.load('../model-input/grl40_NorESM_t2m.npy'), axes=(1,0,2))
    NorESM_prec = np.transpose(np.load('../model-input/grl40_NorESM_prec.npy'), axes=(1,0,2))
    clim_index  = np.load('../model-input/climate_index.npy')
    return ERA5_t2m, ERA5_prec, NorESM_t2m, NorESM_prec, clim_index

### PLOTTING ###################################################################
def initialize_plot(xlon, ylat, z, h, smb):
    trans = ccrs.PlateCarree()
    proj = ccrs.NorthPolarStereo(central_longitude=-44)

    # add coastline of greenland
    # get path to shapefile
    shpfilename = shpreader.natural_earth(resolution='50m',
                                          category='cultural',
                                          name='admin_0_countries')
    # read shapefile
    shpfile = shpreader.Reader(shpfilename)
    # select greenland only
    greenland = [country.geometry for country in shpfile.records() \
                 if country.attributes["NAME_LONG"] == "Greenland"]
    
    fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection = proj), figsize=(10, 6))
    axs = axs.flatten()

    # save artists as variables to access in update func
    cm_z = axs[0].pcolormesh(xlon, ylat, z[0], cmap='terrain',
                             shading='nearest', transform=trans)
    fig.colorbar(cm_z, orientation='horizontal', pad=0.05, extend='min')
    cm_h = axs[1].pcolormesh(xlon, ylat, h[0], cmap='jet',
                             shading='nearest', transform=trans)
    fig.colorbar(cm_h, orientation='horizontal', pad=0.05)
    cm_smb = axs[2].pcolormesh(xlon, ylat, smb[0], cmap='bwr',
                               shading='nearest', transform=trans,
                               norm=mpl.colors.CenteredNorm())
    fig.colorbar(cm_smb, orientation='horizontal', pad=0.05)

    subtitles = ['Elevation [m]', 'Ice thickness [m]', 'Surface mass balance [m/yr]']
    for i, title in enumerate(subtitles):
        axs[i].set_title(title)
        # add greenland coastline
        axs[i].add_geometries(greenland, crs=trans, fc='none', ec='k', alpha=0.8)
        axs[i].gridlines(alpha=.6, ls='--', lw=.5)

    fig.suptitle("Greenland ice sheet model")

    # plot animation in separate window
    plt.ion()
    plt.show()
    plt.pause(.1)

    return fig, cm_z, cm_h, cm_smb

def update_plot(t, fig, cm_z, cm_h, cm_smb, z, h, smb, nt):
    # the first argument will be the next value in frames
    fig.suptitle(f"Greenland ice sheet model - time step {t}/{nt} [yr]")

    # update mesh
    cm_z.set_array(z[t])
    cm_z.norm = mpl.colors.Normalize(vmin=0)
    cm_h.set_array(h[t])
    cm_h.autoscale()
    cm_smb.set_array(smb[t])
    cm_smb.autoscale()
    # https://stackoverflow.com/questions/12822762/pylab-ion-in-python-2-matplotlib-1-1-1-and-updating-of-the-plot-while-the-pro/12826273
    plt.pause(.1)

### SAVE RESULTS ###############################################################
def save_results(seamask, nt, xlon, ylat, z, h, smb):
    # save ocean mask 
    seamask = np.tile(seamask, reps=(nt, 1, 1))
    # save model results
    np.savez(f'{SIM_PATH}/simulation-data.npz',
             z=z, h=h, smb=smb, xlon=xlon, ylat=ylat, nt=nt, seamask=seamask)

### RUN MODEL ##################################################################
def run_model(logger):
    # set up model
    bedrock, surface, seamask, xlon, ylat = import_data()
    nx, ny, dx, dy, nt, dt, A, RHO, G = set_model_parameters(bedrock)
    output_model_parameters_to_logger(nx, ny, dx, dy, dt, nt, logger)
    z, h, smb, u_x, u_y, u_abs, Fmx, Fpx, Fmy, Fpy = initialize_model(nt, nx, ny, bedrock, surface)
    fig, cm_z, cm_h, cm_smb = initialize_plot(xlon, ylat, z, h, smb)
    
    # set up surface mass balance
    if REALISTIC_SMB:
        ERA5_t, ERA5_pr, NorESM_t, NorESM_pr, clim_index = load_climate_data()
    else: # load distance to coast for idealized smb & output smb parameters to logger
        distance = load_distance_to_coast()
        output_smb_params_to_logger(SIM_PATH, logger)
    
    # run model
    for t in range(0, nt-1):

        # compute/update surface mass balance
        if BENCHMARK:
            smb[t] = smb_benchmark(bedrock)
        elif REALISTIC_SMB:
            smb[t] = compute_smb_NorESM(ERA5_t, ERA5_pr, NorESM_t, NorESM_pr, clim_index, t) 
        else: # idealized smb
            smb[t] = compute_smb(z[t], seamask, SIM_PATH, distance, verbose=False)

        for ix in range(0, nx):
            for iy in range(0, ny):
                # fluxes in x-direction [m^2/yr]
                if ix == 0:
                    Fmx[ix, iy] = 0.
                    Fpx[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix+1,iy] - z[t,ix,iy]) / dx)**3 \
                                  * (.5 * (h[t,ix+1,iy]+h[t,ix,iy]))**5 \
                                  *60*60*24*365 # convert to [m^2/yr]
                elif ix == nx-1:
                    Fmx[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy] - z[t,ix-1,iy]) / dx)**3 \
                                  * (.5 * (h[t,ix,iy]+h[t,ix-1,iy]))**5 \
                                  *60*60*24*365
                    Fpx[ix, iy] = 0.
                else:
                    Fmx[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy] - z[t,ix-1,iy]) / dx)**3 \
                                  * (.5 * (h[t,ix,iy]+h[t,ix-1,iy]))**5 \
                                  *60*60*24*365
                    Fpx[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix+1,iy] - z[t,ix,iy]) / dx)**3 \
                                  * (.5 * (h[t,ix+1,iy]+h[t,ix,iy]))**5 \
                                  *60*60*24*365
                
                # fluxes in y-direction [m^2/yr]
                if iy == 0:
                    Fmy[ix, iy] = 0.
                    Fpy[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy+1] - z[t,ix,iy]) / dy)**3 \
                                  * (.5 * (h[t,ix,iy+1]+h[t,ix,iy]))**5 \
                                  *60*60*24*365
                elif iy == ny-1:
                    Fmy[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy] - z[t,ix,iy-1]) / dy)**3 \
                                  * (.5 * (h[t,ix,iy]+h[t,ix,iy-1]))**5 \
                                  *60*60*24*365
                    Fpy[ix, iy] = 0.
                else:
                    Fmy[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy] - z[t,ix,iy-1]) / dy)**3 \
                                  * (.5 * (h[t,ix,iy]+h[t,ix,iy-1]))**5 \
                                  *60*60*24*365
                    Fpy[ix, iy] = -2./5.*A*(RHO*G * (z[t,ix,iy+1] - z[t,ix,iy]) / dy)**3 \
                                  * (.5 * (h[t,ix,iy+1]+h[t,ix,iy]))**5 \
                                  *60*60*24*365
                
                # calculate ice velocity (average fluxes divided by ice thickness)
                # in x-direction
                #u_x[t,ix,iy] = (Fpx[ix,iy]-Fmx[ix,iy]) / (2 * h[t,ix,iy])
                # in y-direction
                #u_y[t,ix,iy] = (Fpy[ix,iy]-Fmy[ix,iy]) / (2 * h[t,ix,iy])
                # compute the absolute ice velocity
                #u_abs[t,ix,iy] = np.sqrt(u_x[t,ix,iy]**2 + u_y[t,ix,iy]**2)
                
                # calculate ice thickness (can't be negative)
                h[t+1,ix,iy] = max(0, h[t,ix,iy] - (Fpx[ix,iy]-Fmx[ix,iy])*(dt/dx) - (Fpy[ix,iy]-Fmy[ix,iy])*(dt/dy) + smb[t,ix,iy]*dt)

                # update elevation
                z[t+1,ix,iy] = bedrock[ix,iy] + h[t+1,ix,iy]

        # add calving
        h[z < 0] = 0

        # update maps with model results
        if t % 50 == 0:
            #print(f"Time step {t} completed")
            update_plot(t, fig, cm_z, cm_h, cm_smb, z, h, smb, nt)
        
    # save model results
    if SAVE_RESULTS:
        # compute last smb, as last index is not reached in loop
        if REALISTIC_SMB:
            smb[nt-1] = compute_smb_NorESM(ERA5_t, ERA5_pr, NorESM_t, NorESM_pr, clim_index, nt-1)
        else: # idealized smb
            smb[nt-1] = compute_smb(z[nt-1], seamask, SIM_PATH, distance, verbose=False)
        save_results(seamask, nt, xlon, ylat, z, h, smb)
        logger.info("Model results saved!\n")

### MAIN #######################################################################

# cProfiler for performance analysis:
#with cProfile.Profile() as pr:
    
# measure programme runtime & logg console output
start_time = time.monotonic()
# set up logger
logger1 = logging_setup()
logger1.info("2D Greenland Ice Sheet Model")
logger1.info("Simulation path: %s", SIM_PATH)
logger1.info("%s\n", datetime.now())
# run model
run_model(logger1)
# print runtime
end_time = time.monotonic()
logger1.info('runtime = %s\n', timedelta(seconds=end_time - start_time))

# print profiling results
#results = pstats.Stats(pr)
#results.sort_stats(pstats.SortKey.TIME)
#results.print_stats()
#results.dump_stats(f'{SIM_PATH}/results.prof')
