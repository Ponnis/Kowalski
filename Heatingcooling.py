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
    
def compute_H2O_cooling(w, r_center):
    """
    Calculates H2O cooling (rotational + v2 vibrational) following 
    Kasting & Pollack (1983) and Hollenbach & McKee formulations.
    """
    Tgas = h.get_temperatures(w)
    dr = r_center[1] - r_center[0]  # Radial cell width array [cm]
    
    n_tot = h.get_number_density(w)
    n_h2o = n_tot*p.X_h2o
    n_h2  = n_tot*p.X_h2
    n_h   = n_tot*p.X_h
    n_he  = n_tot*p.X_he
    
    # Physical Constants
    A0 = 2.3e-2            # s^-1 (Einstein coefficient)
    sigmaH2 = 2.5e-15      # cm^2 (H2 cross-section)
    DopplerWidth = 1.0     # Doppler width parameter
    kB = getattr(c, 'kB', 1.380649e-16)           # erg/K
    Mproton = getattr(c, 'Mproton', 1.6726219e-24) # g

    integrand = 0.5 * (dr * n_h2o[:-1] + dr * n_h2o[1:])
    N_H2O = np.zeros_like(Tgas)
    N_H2O[-1] = 0.5 * dr * n_h2o[-1]
    N_H2O[:-1] = N_H2O[-1] + np.cumsum(integrand[::-1])[::-1]

    etaJT = 7.7 * (Tgas / 1000.0)**0.5
    tauT = (4.0 * N_H2O * A0) / (etaJT * Tgas * 1.18e7 * DopplerWidth * (17.0**2.0))
    
    vT = np.sqrt(8.0 * kB * Tgas / (np.pi * 2.0 * Mproton))
    nCR = (4.0 * Tgas / 17.0) * A0 / (sigmaH2 * vT)
    
    Ctau = tauT * np.sqrt(2.0 * np.pi * np.log(2.13 + (tauT / 2.718)**2.0))
    Ym = np.log(1.0 + Ctau / (1.0 + 10.0 * (nCR / n_h2)))
    
    L_H2O = (
        (2.0 * kB * (Tgas**2.0) * A0 / (n_h2 * 17.0))
        * (2.0 + Ym + 0.6 * (Ym**2.0))
        / (1.0 + Ctau + (nCR / n_h2) + 1.5 * np.sqrt(nCR / n_h2))
    )
    
    Q_rot = n_h2 * n_h2o * L_H2O

    E_kB = 2294.0         # K
    hnu = 3.167e-13        # erg (photon energy at 6.27 µm)
    A10 = 18.0            # s^-1
    sigma_vib = 1.0e-16   # cm^2
    
    sigmaN = N_H2O * sigma_vib
    escapeprob_vib = 0.5 / (1.0 + sigmaN * np.sqrt(2.0 * np.pi * np.log(2.13 + sigmaN**2.0)))
    
    # Collisional de-excitation rate (H2, H, and He collisions)
    T_factor = np.sqrt(Tgas / 300.0)
    kM_deex = (
        3.0e-12 * T_factor * n_h2 +
        1.0e-11 * T_factor * n_h +
        1.0e-12 * T_factor * n_he
    )
    
    # NLTE population & vibrational cooling rate
    n_H2O_vib = (n_h2o * np.exp(-E_kB / Tgas) * kM_deex) / (kM_deex + A10 * escapeprob_vib)
    Q_vib = hnu * A10 * n_H2O_vib * escapeprob_vib

    # Total cooling
    Q_h2o = Q_rot + Q_vib
    return Q_h2o

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

def compute_H3j_cooling(w):
    """
    Computes H3+ IR cooling in the upper thermosphere.
    Uses the thin-limit analytical parameterization (e.g., Miller et al. 2013, 
    Glover & Jappsen 2007) for the fundamental 3-4 micron vibrational modes.
    """
    Tgas = h.get_temperatures(w)
    n_tot = h.get_number_density(w)
    
    # H3+ number density [cm^-3]
    # (Uses p.X_h3plus if defined, or defaults to a fraction of n_e / n_tot)
    X_h3j = getattr(p, 'X_h3j', 1e-6)
    n_h3plus = n_tot * X_h3j

    # Effective parameters for the v2 fundamental vibrational manifold
    # Cooling rate per H3+ ion: L_H3+ ~ A * hnu * exp(-E / kB / T)
    E_eff_kB = 3280.0       # Effective excitation energy in Kelvin (~3.8 µm)
    L_0 = 6.0e-20           # Base cooling efficiency [erg s^-1 molecule^-1]

    # Radiative cooling rate per molecule in the optically thin regime
    L_h3plus = L_0 * np.exp(-E_eff_kB / Tgas)

    # Volumetric cooling rate [erg cm^-3 s^-1]
    Q_h3plus = n_h3plus * L_h3plus
    return Q_h3plus

import numpy as np

def compute_IR_heating(w, r_center):
    """
    Calculates volumetric near-IR solar heating [erg/cm^3/s] from 
    H2O, CO2, and CH4 absorption bands.
    
    Returns:
        Q_IR_heat_tot (ndarray): Total volumetric IR heating rate [erg/cm^3/s]
        Q_IR_heats (list): List of individual heating components [H2O, CO2, CH4]
    """
    # 1. State Extraction & Setup
    Tgas = h.get_temperatures(w)
    n_tot = h.get_number_density(w)
    dr = r_center[1] - r_center[0]  # Cell width [cm]

    # Species mixing ratios (fallback defaults if not defined in p)
    X_h2o = getattr(p, 'X_h2o', 4.0e-4)
    X_co2 = getattr(p, 'X_co2', 2.0e-5)
    X_ch4 = getattr(p, 'X_ch4', 1.0e-6)

    n_h2o = n_tot * X_h2o
    n_co2 = n_tot * X_co2
    n_ch4 = n_tot * X_ch4

    # 2. Top-down column density calculation [cm^-2]
    def calc_column_density(n_spec):
        integrand = 0.5 * (dr * n_spec[:-1] + dr * n_spec[1:])
        N = np.zeros_like(Tgas)
        N[-1] = 0.5 * dr * n_spec[-1]
        N[:-1] = N[-1] + np.cumsum(integrand[::-1])[::-1]
        return N

    N_H2O = calc_column_density(n_h2o)
    N_CO2 = calc_column_density(n_co2)
    N_CH4 = calc_column_density(n_ch4)

    # 3. Radiation Parameters & Band Fluxes
    eta_heat   = getattr(p, 'eta_heat', 0.15)         # Thermalization efficiency
    F_nir_h2o  = getattr(p, 'F_nir_h2o', 3.2e7)       # erg cm^-2 s^-1
    F_nir_co2  = getattr(p, 'F_nir_co2', 8.0e6)       # erg cm^-2 s^-1
    F_nir_ch4  = getattr(p, 'F_nir_ch4', 1.2e7)       # erg cm^-2 s^-1

    # Cross-sections [cm^2]
    sigma_h2o  = getattr(p, 'sigma_nir_h2o', 1.2e-21)
    sigma_co2  = getattr(p, 'sigma_nir_co2', 3.5e-21) * np.sqrt(Tgas / 300.0)
    sigma_ch4  = getattr(p, 'sigma_nir_ch4', 2.0e-21)

    # 4. Band Optical Depths
    tau_h2o = N_H2O * sigma_h2o
    tau_co2 = N_CO2 * sigma_co2
    tau_ch4 = N_CH4 * sigma_ch4

    # 5. Volumetric Heating Rates [erg/cm^3/s]
    Q_h2o_heat = eta_heat * sigma_h2o * n_h2o * F_nir_h2o * np.exp(-tau_h2o)
    Q_co2_heat = eta_heat * sigma_co2 * n_co2 * F_nir_co2 * np.exp(-tau_co2)
    Q_ch4_heat = eta_heat * sigma_ch4 * n_ch4 * F_nir_ch4 * np.exp(-tau_ch4)

    Q_IR_heat_tot = Q_h2o_heat + Q_co2_heat + Q_ch4_heat
    Q_IR_heats = [Q_h2o_heat, Q_co2_heat, Q_ch4_heat]

    return Q_IR_heat_tot, Q_IR_heats