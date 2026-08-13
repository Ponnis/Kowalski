import numpy as np
import Constants as c
import Parameters as p
import Helpers as h

def Q_fit(T_eval, coeffs):
    """Evaluate the fitted Q(T). Vectorized, works for scalars or arrays."""
    T_temp = np.clip(T_eval,a_min = 0, a_max = 1000)
    return 10.0 ** np.polyval(coeffs, np.log10(T_temp))
    
def compute_CO_cooling(w):
    T = h.get_temperatures(w)
    n = h.get_number_density(w)
    
    a6 = -3.3949449463e+01
    a5 =  5.2590281005e+02
    a4 = -3.3728829238e+03
    a3 =  1.1461731300e+04
    a2 = -2.1769322920e+04
    a1 =  2.1931247116e+04
    a0 = -9.1974838734e+03
    
    coeffs = [a6,a5,a4,a3,a2,a1,a0]
    Q_co_sr = Q_fit(T, coeffs)
    if(p.debug):print(f'Qfit = {Q_co_sr}')
    if(p.debug):print(f'n_co = {n*p.X_co}')
    Q_co =  4 * c.PI * Q_co_sr * n*p.X_co
    return Q_co

def compute_CH4_cooling(w):
    """
    Simple mimic of LTE CH4 cooling from the exomol cooling function.
    """
    
    Tgas = h.get_temperatures(w)
    n = h.get_number_density(w)
    
    n_ch4 = p.X_ch4*n # mixing ratio of 1 ppm, just a placeholder for ish JPT value
    Q_CH4 = np.zeros_like(Tgas)
    
    for i in range(len(Tgas)):
        
        if Tgas[i] > 1500:
            Q_CH4[i] = 2.98128643e-13
        elif Tgas[i] < 75:
            Q_CH4[i] = 6.1e-31*(Tgas[i]**3.56)*np.exp(-36.28/Tgas[i])
        else:
            Q_CH4[i] = 6.81e-19*(Tgas[i]**1.91)*np.exp(-1515.26/Tgas[i])
            
    Q_CH4 = Q_CH4*n_ch4*2*c.PI
        
    return Q_CH4
    
    
def compute_xuv_heating(w, r_center):
    """
    Single-band Beer-Lambert XUV heating.
    Returns heating in [erg/cm^3/s] at each cell centre.
    This is not in any way physical, just to test perturbations.
    """
    
    eta       = 1                      # heating efficiency
    sigma_xuv = 2.5e-20                     # cm^2, H photoionisation cross-section
    F_top     = p.solar_XUV / (p.a**2)          # flux at top of atmosphere [erg/cm^2/s]
    floor = 1e-25
    
    # Number density of absorbers [cm^-3]
    n_abs = w[p.irho] / p.m_bar
    
    # Optical depth integrated downward from the top
    dr   = r_center[1] - r_center[0]
    tau  = np.zeros(p.n_cells)
    for i in range(p.n_cells - 2, -1, -1):
        tau[i] = tau[i+1] + sigma_xuv * n_abs[i+1] * dr
    #print(f"tau=1 at cell {np.argmin(np.abs(tau - 1.0))}, r = {r_center[np.argmin(np.abs(tau-1.0))]/c.R_JUPITER:.4f} R_J")
    # Volumetric heating rate [erg/cm^3/s]
    
    Q = eta * sigma_xuv * n_abs * F_top * np.exp(-tau)
    
    for i in range(len(Q)):
        if Q[i]<floor:
            Q[i] = 0
    
    return Q

def compute_metal_line_cooling(w):
    # All from Huang et al 2017
    T = h.get_temperatures(w)
    T4 = T/1e4
    #g = 2J+1
    g_u = 4 # Using the more energetic one because Idk here
    g_l = 2
    C_ul_mg2 = 2.6e8/(2.2e14)*np.sqrt(4000/T)
    C_ul_na2 = 6.16e7/(4.4e14)*np.sqrt(4000/T)
    C_lu_mg2 = (g_u/g_l)*C_ul_mg2*np.exp(-4.422/c.K_B_cgs*T)
    C_lu_na2 = (g_u/g_l)*C_ul_na2*np.exp(-2.104/c.K_B_cgs*T)
    Q_mg1 = (3.4*1e-19*T**0.18*np.exp(-5.04/T4)) + 1e-16*np.exp(-3.15/ T4)/(254 + 3.0*1e-8*p.n_e ) # 2852 + 4571
    Q_mg2 = 2*(7.1e-12*C_lu_mg2)
    Q_na2 = 3.4e-12*C_lu_na2
    Q_k1  = 3.7e-19*T**0.18*np.exp(-1.87/T4)
    Q_ff  = 1.85e-27*np.sqrt(T)
    return Q_mg1,Q_mg2,Q_na2,Q_k1,Q_ff