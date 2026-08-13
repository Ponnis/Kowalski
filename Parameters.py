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
a = 1  # au
solar_XUV = 2334 # erg/cm²/s  e9 actually
eta_cool = 1e-1
eta_heat = 20
gamma = 5/3
irho = 0
im = 1
ie = 2
iv = 1
iP = 2
n_timesteps = 100000

### CHEMISTRY ###
n_e = 1e9 # Electron number density
X_co = 0.004
X_h2o = 0.0031
X_ch4 = 1e-5
### HEATING & COOLING ###