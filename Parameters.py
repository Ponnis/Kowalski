import Constants as c
###### PARAMETERS ######
n_cells = 1000
r_min = 13.93*c.R_EARTH # cm
r_max = r_min*1.1 # cm
mass = 89*c.M_EARTH # g
n_ghost = 2
T = 700 # K
m_bar = 2.4*c.M_HYDROGEN # Mean molecular mass in g
P_max = 1e-3*c.BAR_TO_CGS # Dyne/cm²
CFL = 0.5
debug = False
a = 0.045  # au
gamma = 5/3
irho = 0
im = 1
ie = 2
iv = 1
iP = 2
n_timesteps = 100000
RCEstart = False
plot_freq = 1
### CHEMISTRY ###
n_e = 1e9 # Electron number density
X_co = 0.004
X_h2  = 0.86
X_he  = 0.14
X_h   = 1e-5
X_h2o = 0.0031
X_ch4 = 1e-20
X_h3j = 1e-5
### HEATING & COOLING ###
solar_XUV = 2334 # erg/cm²/s  e9 actually
eta_cool = 1e-1
eta_heat = 1
### XUV ###
spec_path = "Data/Spec/WASP39.dat"
spec_distance_au = 1.0   # Distance at which the spectrum is specified [AU]
xuv_emin = 13.6          # Lower photon energy [eV]
xuv_emax = 12400.0       # Upper photon energy [eV]

eta_xuv = 0.15           # Fraction of absorbed energy converted to gas heat
sigma_xuv = 2.5e-21      # Effective grey cross section per particle [cm²]