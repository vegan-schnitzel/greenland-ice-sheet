#####################
# 1D Ice Flow Model #
# Robert Wright     #
# 02/2024           #
#####################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='ticks', color_codes=True)

# set model parameters
L = 1000                  # domain size
nx = 200                  # number of grid points
x = np.linspace(0, L, nx) # spatial dimension
dx = L / (nx - 1)         # distance between grid points [m]
print("dx =", dx, "m")

nt = 400 # number of time steps
dt = 1   # time step [yr]
print("dt =", dt, "yr")

# set simulation parameters
A   = 1e-20    # flow parameter [s^-1 Pa^-3]
rho = .92 *1e3 # ice density [kg m^-3]
g   = 9.81     # gravitational acceleration [m s^-2]


# create surface mass balance
def init_smb(x, half_range=L/8):
    # use step function
    condition = np.logical_and(x >= L/2-half_range, x <= L/2+half_range)
    return np.where(condition, .01, -.01) # ! smb is changed

#plt.plot(x, init_smb(x), label="smb")

# initial ice thickness [m]
h = np.zeros((nt, nx))
# and ice fluxes [m^2/s]
Fm = np.zeros(nx)
Fp = np.zeros(nx)
# and ice speed [m/yr]
speed = np.zeros((nt, nx))
# surface mass balance [m/yr]
SMB = init_smb(x)
# alternative smb:
#SMB = np.linspace(.1, -.1, nx)

# calculate ice flux due to deformation
def ice_flux_minus(t,n):
    flux = -2./5.*A*(rho*g * (h[t,n] - h[t,n-1]) / dx)**3 \
           * (.5 * (h[t,n]+h[t,n-1]))**5
    return flux *60*60*24*365 # convert to [m^2/yr]

def ice_flux_plus(t,n):
    flux = -2./5.*A*(rho*g * (h[t,n+1] - h[t,n]) / dx)**3 \
           * (.5 * (h[t,n+1]+h[t,n]))**5
    return flux *60*60*24*365 # convert to [m^2/yr]

# run simulation
for t in range(0,nt-1):
    for n in range(0,nx):
        # calculate ice fluxes
        if n == 0:
            Fm[n] = 0.
            Fp[n] = ice_flux_plus(t,n)
        elif n == nx-1:
            Fm[n] = ice_flux_minus(t,n)
            Fp[n] = 0.
        else:
            Fm[n] = ice_flux_minus(t,n)
            Fp[n] = ice_flux_plus(t,n)
        
        # get ice speed (average of fluxes divided by ice thickness)
        speed[t,n] = (Fp[n]+Fm[n])/2 / h[t,n] if h[t,n] != 0 else 0
        # calculate Courant number
        C = dt / dx * speed[t,n]
        if C > 1:
            print(f"Courant number {np.round(C,1)} > 1 at t={t} and n={n}; ice speed={np.round(speed[t,n],1)}m/yr")
        
        # update height
        h[t+1,n] = max(0, h[t,n] - (Fp[n]-Fm[n]) * (dt/dx) + SMB[n]*dt)


# plot ice thickness
fig, ax = plt.subplots()
step_size = nt // 5  # calculate step size to get 10 data points
for yr in range(0, nt, step_size):
    ax.plot(x, h[yr,:], label=f"t={np.round(yr*dt,1)}yr")
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("ice thickness [m]")
ax.set_title("1D Ice Flow Model")
fig.savefig("figs/1d-ice-flow.png", dpi=300, bbox_inches='tight')
plt.draw()
