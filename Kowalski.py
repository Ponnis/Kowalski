import numpy as np
import pandas as pd
import polars as pl
import Constants as c
import Parameters as p
import Helpers as h
import Heatingcooling as hc
import matplotlib.pyplot as plt
import matplotlib.image as image

def compute_heating_cooling(w, r_center):
    """
    Collection method for all the heating and cooling effects
    """
    
    Q_xuv = hc.compute_xuv_heating(w, r_center)
    Q_CH4 = hc.compute_CH4_cooling(w)
    Q_CO = hc.compute_CO_cooling(w)
    Q_mg1, Q_mg2, Q_na2, Q_k1, Q_ff = hc.compute_metal_line_cooling(w)
    Q_cools = [Q_mg1, Q_mg2, Q_na2, Q_k1, Q_ff, Q_CH4, Q_CO]
    Q_heats = [Q_xuv]
    Q_total = p.eta_heat*sum(Q_heats) - p.eta_cool*sum(Q_cools)
    
    return Q_total, Q_cools, Q_heats
    
def sound_speed(P, rho):
    """
    Calculates the sound speed Sqrt(p.gamma*P/rho)
    """
    
    return np.sqrt(p.gamma*P/rho)

def calc_face_fluxes(U, w, phi, r_center):
    """
    Calculates the fluxes at the cell faces by 
    """
    
    F_face = np.zeros((3, p.n_cells + 1))

    # Bottom face (boring)
    F_face[:, 0] = calc_flux(U[:, 0], U[:, 1], w[:, 0], w[:, 1])
    
    for i in range(1, p.n_cells):
        dphi = phi[i] - phi[i-1]           # cell-centred derivative

        # Construct left face values according to 
        P_L   = w[p.iP,  i-1] - w[p.irho, i-1] * dphi / 2
        rho_L = w[p.irho, i-1]
        v_L   = w[p.iv,   i-1]
        E_L   = P_L / (p.gamma - 1) + 0.5 * rho_L * v_L**2
        U_L   = np.array([rho_L, rho_L * v_L, E_L])
        w_L   = np.array([rho_L, v_L, P_L])

        P_R   = w[p.iP,  i]   + w[p.irho, i]   * dphi / 2
        rho_R = w[p.irho, i]
        v_R   = w[p.iv,   i]
        E_R   = P_R / (p.gamma - 1) + 0.5 * rho_R * v_R**2
        U_R   = np.array([rho_R, rho_R * v_R, E_R])
        w_R   = np.array([rho_R, v_R, P_R])

        F_face[:, i] = calc_flux(U_L, U_R, w_L, w_R)
    
    # Top boundary face — MK16 linear reconstruction + transmissive outflow
    dr       = r_center[1] - r_center[0]
    phi_ext  = -c.G_CGS * p.mass / (r_center[-1] + dr)   # one ghost cell beyond top
    dphi_top = phi_ext - phi[-1]                         # positive: going outward
    
    P_top   = w[p.iP,  -1] - w[p.irho, -1] * dphi_top / 2
    rho_top = w[p.irho, -1]
    v_top   = w[p.iv,   -1]
    E_top   = P_top / (p.gamma - 1) + 0.5 * rho_top * v_top**2
    w_top   = np.array([rho_top, v_top,          P_top])
    U_top   = np.array([rho_top, rho_top * v_top, E_top])
    F_face[:, p.n_cells] = calc_flux(U_top, U_top, w_top, w_top)  # transmissive outflow, basically what should leave the sim
        
    return F_face    

def calc_star_flux(U, w, S_k, S_star, F_k):
    """
    Calculates the flux at the middle barrier between two cells.
    """
    
    U_star = w[p.irho]*((S_k - w[p.iv])/(S_k - S_star))*np.array([1,S_star,(U[p.ie]/w[p.irho] + (S_star- w[p.iv])*(S_star + w[p.iP]/(w[p.irho]*(S_k-w[p.iv]))))])
    F_star = F_k + S_k*(U_star - U)
    return F_star

def calc_flux(U_L, U_R, w_L, w_R):
    """
    Calculates the flux at the border between two cells U_L (the lower cell) and U_R (the upper cell).
    This is done by finding the wave speeds from the left, right, and middle, which are then compared
    against eachother to determine the "true" direction of the flux.
    """
    true_flux = 0
    # calculate local sound speed
    cs_L = sound_speed(w_L[p.iP], w_L[p.irho])
    cs_R = sound_speed(w_R[p.iP], w_R[p.irho])
    
    # find wave speeds
    S_L = min(w_L[p.iv]-cs_L, w_R[p.iv]-cs_R)
    S_R = max(w_L[p.iv]+cs_L, w_R[p.iv]+cs_R)
    
    # find the middle wave speed    
    S_star = (w_R[p.iP] - w_L[p.iP] + w_L[p.irho]*w_L[p.iv]*(S_L-w_L[p.iv]) -  w_R[p.irho]*w_R[p.iv]*(S_R-w_R[p.iv]))/(w_L[p.irho]*(S_L - w_L[p.iv])- w_R[p.irho]*(S_R - w_R[p.iv]))# no idea why this abomination is called star still
    
    # Get the left and right fluxes
    f_L_rho = U_L[p.im] 
    f_L_m = U_L[p.im]**2.0 / U_L[p.irho] + w_L[p.iP] 
    f_L_e = U_L[p.im] / U_L[p.irho] * ( U_L[p.ie] + w_L[p.iP] )
    f_R_rho = U_R[p.im] 
    f_R_m = U_R[p.im]**2.0 / U_R[p.irho] + w_R[p.iP] 
    f_R_e = U_R[p.im] / U_R[p.irho] * ( U_R[p.ie] + w_R[p.iP] )
    
    F_L = np.array([f_L_rho, f_L_m, f_L_e])
    F_R = np.array([f_R_rho, f_R_m, f_R_e])
    if(S_L >= 0):
        # It's supersonic right, and the face is entirely in the left state ezpz
        true_flux = F_L
        
    elif(S_R <= 0):
        # It's supersonic left, and the face is entirely in the right state ezpz
        true_flux = F_R
        
    elif(S_L <= 0 <= S_star):
        # The face is in the middle region on the left side of the contact wave
        true_flux = calc_star_flux(U_L, w_L, S_L, S_star, F_L)
        
    elif(S_star <= 0 <= S_R):
        # The face is in the middle region on the right side of the contact wave
        true_flux = calc_star_flux(U_R, w_R, S_R, S_star, F_R)
        
    return true_flux

def conservative_to_primitive(U,r):
    """
    Converts the array of conserved quantities (U) to the primitive ones (w).
    """
    
    rho = U[p.irho]
    v = U[p.im] / rho
    phi = -c.G_CGS*p.mass/r
    # E = P/(p.gamma - 1) + 0.5 * rho * v^2  => P = (E - 0.5 * rho * v^2) * (p.gamma - 1)
    P = (U[p.ie] - 0.5 * rho * v**2 ) * (p.gamma - 1)
    
    return np.array([rho, v, P])

def primitive_to_conservative(w, phi, r):
    """
    Calculates the conserved variables (U, F, S) from the primitive (w) (in addition to phi and radius).
    """
    
    rho = w[0]
    v = w[1]
    P = w[2]
    E = P/(p.gamma-1) + 1/2*rho*v**2
    u1 = rho
    u2 = v*rho
    u3 = E
    f1 = rho*v
    f2 = rho*v**2 + P
    f3 = (E+P)*v
    s1 = np.zeros(p.n_cells)
    s2 = -rho
    s3 = -rho*v
    U = np.array([u1,u2,u3]) # Conserved variables
    F = np.array([f1,f2,f3]) # Fluxes
    dphidr = calc_dphi_dr(phi, r)
    S = np.array([s1,s2,s3])*dphidr # Source
    
    return U,F,S


def calc_dt(r_center, w):
    """
    Calculates the size of the timesteps according to local sound speeds and wind speed.
    """
    
    dr = r_center[1] - r_center[0]
    cs = sound_speed(w[p.iP], w[p.irho])
    
    # Maximum speed is fluid velocity + sound speed
    v_max = np.max(np.abs(w[p.iv]) + cs)
    
    dt = p.CFL * dr / v_max
    return dt
    
def plot_initial_conditions(P,temp,n):
    """
    Saves a plot of the initial conditions to Kowalski_init.png
    """
    
    P = P/c.BAR_TO_CGS
    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(n, P, label = "Pressure [bar]", c = "navy", lw = 2.5)
    ax.set_xlabel("Number density [cm^-3]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax1 = ax.twiny()
    ax1.plot(temp, P, label = "Temperature [K]", c = "maroon", lw = 2.5)
    ax1.set_xlabel("Temperature [K]")
    plt.title("Kowalski initial conditions")
    plt.savefig("Plots/Kowalski_init.png")
    return


def calc_face(x):
    """
    Calculates the cell face quantity of a given central quantity. E.g x_i{+1/2}
    """
    
    face = np.zeros(p.n_cells+1)
    
    for i in range(1, p.n_cells):
        face[i] = (x[i-1] + x[i]) / 2
    face[0] = 2*x[0] - x[1] # linear extrapolation
    
    return face

def populate_primitive(rho, v, P):
    """
    Simply just for populating the primitive array, just here for compartmentalisation.
    """
    
    w = np.array([rho, v, P])
    
    return w

def initialise_r_center():
    r_faces = np.linspace(p.r_min, p.r_max, p.n_cells + 1)  # n+1 faces for n cells since bottom cell face doesn't count
    r = 0.5 * (r_faces[:-1] + r_faces[1:])            # midpoints
    return r

def initialise_v():
    """
    Initialises the velocity array. 
    Simply just a bunch of zeroes since we assume hydrostatic initial condition.
    """
    
    # Assume initial hydrostatic condition
    v = np.full(p.n_cells, 0)
    
    return v

def initialise_T(isothermal=True):
    """
    Initialises an isothermal temperature profile. The optional parameter is for future use.
    """
    
    
    temp = 0
    if isothermal:
        temp = np.full(p.n_cells, p.T) 
    return temp                     

def initialise_mbar():
    """
    Initialises a constant mean molecular mass profile according to the given parameter.
    """
    
    mbar = np.full(p.n_cells, p.m_bar)
    return mbar

def initialise_wellbalanced(r_center, temp, mbar):
    """
    We integrate equation 
    """
    mbar_face = calc_face(mbar)
    temp_face = calc_face(temp)
    
    phi = -c.G_CGS*p.mass/r_center # Gravitational potential cell centered
    P = np.zeros(p.n_cells)
    
    P[0] = p.P_max
    for i in range(1,p.n_cells):
        if p.debug:
            print(f'P_i-1 = {P[i-1]}, mbar_face = {mbar_face[i]}, temp_face = {temp_face[i]}, phi_i-1 - phi_i= {phi[i-1]-phi[i]}')
        P[i] = P[i-1]*np.exp(mbar_face[i]/(c.K_B_cgs*temp_face[i])*(phi[i-1]-phi[i]))
        
    rho = P*mbar/(c.K_B_cgs*temp)
    n = P/(c.K_B_cgs*temp)
    
    return P, rho, n, phi

def calc_dphi_dr(phi, r_center):
    """
    Calculates the derivative of the gravitational potential with respect
    to the radius r on the faces of the cell dphi/dr_i{+1/2}.
    """
    
    dr = r_center[1] - r_center[0]
    dphidr = np.zeros(p.n_cells)
    dphidr[1:-1] = (phi[2:] - phi[:-2]) / (2 * dr)   # centred, cell-centre phi
    dphidr[0]    = (phi[1]  - phi[0])  / dr            # one-sided at bottom
    dphidr[-1]   = (phi[-1] - phi[-2]) / dr            # one-sided at top
    
    return dphidr
    

def gravitational_potential(r):
    """
    Calculates the gravitational potential at the cell centers.
    """
    
    g_r = c.G_CGS*p.mass/r**2
    return g_r
    
def check_well_balanced_residual(r_center, P, rho, phi, temp, mbar):
    """
    Checks the exact discrete well-balanced condition between adjacent cells.
    Matches the exponential hydrostatic initialization profile.
    """
    n = len(r_center)
    max_error = 0.0
    
    mbar_face = calc_face(mbar)
    temp_face = calc_face(temp)
    
    # Check the balance step-by-step from cell to cell
    for i in range(1, n):
        
        # What the pressure SHOULD be based on the discrete equation
        P_expected = P[i-1] * np.exp(mbar_face[i] / (c.K_B_cgs * temp_face[i]) * (phi[i-1] - phi[i]))
        if (p.debug): print(P_expected)
        # Relative error between actual and expected
        error = np.abs(P[i] - P_expected) / P[i]
        
        if error > max_error:
            max_error = error
            
    print("--- Well-Balanced Verification ---")
    print(f"Maximum relative cell-to-cell error: {max_error:.2e}")
    
    if max_error < 1e-12:
        print("SUCCESS: The discrete initial condition is perfectly balanced at this resolution!")
    else:
        print("WARNING: Mismatch detected. Ensure reconstruction fluxes match this exact stencil.")
        
    return max_error

def apply_boundary_conditions(U, phi, r_center):
    w_real  = conservative_to_primitive(U[:, p.n_ghost:p.n_ghost+1], r_center[p.n_ghost:p.n_ghost+1])
    P_real  = w_real[p.iP,   0]
    rho_real = w_real[p.irho, 0]

    P_prev = P_real
    for i in range(p.n_ghost - 1, -1, -1):
        P_ghost   = P_prev + rho_real * (phi[i+1] - phi[i])
        rho_ghost = P_ghost * p.m_bar / (c.K_B_cgs * p.T)   # isothermal, OK for BC
        E_ghost   = P_ghost / (p.gamma - 1)
        U[p.irho, i] = rho_ghost
        U[p.im,   i] = 0.0
        U[p.ie,   i] = E_ghost
        P_prev     = P_ghost
    return U

def update_S(w, dphidr, Q=None):
    """
    Updates the conservative fluxes according to the primitive variables and dphi/dr.
    """
    
    S = np.array([
            np.zeros(p.n_cells),
            -w[p.irho] * dphidr,
            -w[p.irho] * w[p.iv] * dphidr
        ])
    if Q is not None:
        S[p.ie] += Q
    return S

def update_U(U, F_face, S, dt, dr):
    """
    Updates the conserved quantities for one timestep
    """
    
    U_new = np.copy(U)
    for i in range(p.n_ghost, p.n_cells):
        U_new[:, i] = U[:, i] \
                    - (dt/dr) * (F_face[:, i+1] - F_face[:, i]) \
                    + dt * S[:, i]
    return U_new

def call_kowalski():
    print("Calling Kowalski to analyse...")
    print("""                                                                                                    
                                                                                                    
                                                                                                    
                                                                                                    
                                      ....-%%%%#:..                                                 
                                      ..%%%%%%%###*...                                              
                                      :%%%%%%%%%####*.                                              
                                    ..%%%###%%%%######...                                           
                                    ..%####*-:---::#%%%:                                            
                                    .=##%#*-=====--::%%%:..                                         
                                    .+#%%%=-=====----:%*#..                                         
                                    ..=+=+--=+-:*---:::::*.                                         
                                    ..*#-*-:=-+*-+--::::::#...                                      
                                    ..=:##=+-=###:---:::::=+..                                      
                                    ..==##+*#-::::--:::::::#:.                                      
                                     .=---#**##********::::%#...                                    
                                    ..=##*+*****++****##:::%%#..                                    
                                    ..*+++++++**##*=::---::@%%:.                                    
                                    ..+-=++*###*==----:::::*#%%.                                    
                                    . =%%%%%#**+==---:::...+%%%= .                                  
                                    ..-=+*#%#*+==---:::::..%%%%%.                                   
                                    ..---::::---==--::::..-%%%%%:..                                 
                                    .::::::.::------::::..%%%%%%%..                                 
                         ............:::......::::::::....%%%%%#%-.                                 
                         .....:-=+++++++********+++=-:...:##%###%%.                                 
                  ....-===================------::...:-:..######%%%..                               
                  ..+++++++++++++================------::.-+-*#%@%%%.                               
                  .=++****++++++++++===============+*+=-:..++#%%@@%%#                               
                  ..++--------------------------------==-:.*#%%%%@%%%=..                            
                    :+----------::::::---:::-:::------+=--.-#%%%%@%#%%..                            
                   ..=+:--------:::::--:-:::-:::::----+=--.::%%%%@@%%%%.                            
                    ..++---------::::::---::::::::----==--.::-%%%%@%#%%:                            
                    ...+---------::::-::--:::::::::----==-.#=-=%%%@@#%%#..                          
                       :+:---------:------:::::::::----+=-.###++%%@@##%%..                          
                       .=+---------------:----:::::----+=-.%%%%##%@@%#%%..                          
                       ..+=-----------------:-----:----+=-.#%%%%%#@@%#%%:.                          
                       ..%+----------------------------+=-.#%%%%%%%%%%%%..                          
                       .=%%%%%%%%#*=-------------------+=-:*@@@@%%%%%%%%..                          
                       .-@%%%%%%%%%%%@*----------------=+=:+@@@@@@@@%%%*..                          
                       ..#@@%%@@@@@*--------------------+=-=*#@@@@@@@@%.                            
                        . *@@@@*------------------------+=--++**%@@@@@..                            
                         ... :+-------------------------+==--===++===%..                            
                            ..==------------------------+==----------+..                            
                              .+-=----------------------+==-::-----:::..                            
                              .-+=::-=------------------++=-:::--::::::                             
                              .--:::::-=+*+=-:::-=------++=-:::--:::::..                            
                              .--:::::..........:=++*+=-+++:::---::::-..                            
                              .--::::...............::::::::::---::::-..                            
                              .--::::................::::::::-----:::-..                            
                              .=-::::...............:::::::-------:---..                            
                               :--::::.............::::::::----------:.                             
                              ..--:::::..........::::::::------------...                            
                               .=--:::::....:..:::::::::-------------                               
                              ...---::::::::::::::::::::------------:                               
                               ..=---:::::::::::::::::--------------.                               
                               ...=----:::::::::::::----------------.                               
                                 ..=------::::::::--------======--=..                               
                            ......-#==------------------===========..                               
                            -**######+==-------------===========+=...                               
                         .:############+====--------========++++#*:..                               
                       ..******##%######=:++====----=====..%%%########=....                         
                       .... . .-***#+.....  ...::::... ...:%%%############=.....                    
                            . .........       . ...     ...%#####*****####**+...                    
                                                           .***#:+*+++***-......                    
                                                           ..=++*...:*++*++..                       
                                                           ....:+.......:++=.                       
                                                                ...     . ...                       
                                                                                                    
                                                                                                    
    """)
    return

def modify_kowalski(img_kowalski, U, w, phi, r_center):
    """
    Kowalski remains static at hydrostatic equilibrium.
    As the residual grows (flow develops), rows shift horizontally
    and wash toward red, proportional to local departure from balance.
    """
    dr     = r_center[1] - r_center[0]
    dphidr = calc_dphi_dr(phi, r_center)

    # Centred pressure gradient at every cell
    dPdr       = np.zeros(p.n_cells)
    dPdr[1:-1] = (w[p.iP, 2:] - w[p.iP, :-2]) / (2 * dr)
    dPdr[0]    = (w[p.iP, 1]  - w[p.iP, 0])   / dr
    dPdr[-1]   = (w[p.iP, -1] - w[p.iP, -2])  / dr

    # Relative residual, clipped to [0, 1]
    grav         = np.abs(w[p.irho] * dphidr)
    grav         = np.where(grav > 0, grav, 1.0)
    residual     = np.clip(np.abs(dPdr + w[p.irho] * dphidr) / grav, 0.0, 1.0)

    height, width = img_kowalski.shape[:2]
    n_ch          = img_kowalski.shape[2] # Ignore
    img_mod       = img_kowalski.copy().astype(float)
    vmax          = img_mod.max()          # 1.0 for float PNG, 255 for uint8
    max_shift     = max(1, width // 10)   # max horizontal pixel shift

    for row in range(height):
        # row 0 = top of image = top of atmosphere = high cell index
        cell_idx = int((1.0 - row / height) * (p.n_cells - 1))
        cell_idx = np.clip(cell_idx, 0, p.n_cells - 1)
        r = float(residual[cell_idx])

        # Horizontal shift: rows with large residual slide sideways
        shift = int(r * max_shift)
        if shift:
            img_mod[row] = np.roll(img_mod[row], shift, axis=0)

        # Red wash: boost R, suppress G and B
        img_mod[row, :, 0] = np.clip(img_mod[row, :, 0] + r * vmax * 0.8,  0, vmax)
        img_mod[row, :, 1] = np.clip(img_mod[row, :, 1] * (1.0 - 0.6 * r), 0, vmax)
        img_mod[row, :, 2] = np.clip(img_mod[row, :, 2] * (1.0 - 0.6 * r), 0, vmax)
        # channel 3 (alpha, if present) left untouched

    return img_mod.astype(img_kowalski.dtype)

def analysis(t_max, plot_every=1):
    """
    It's exactly what it sounds like lmao
    """
    
    im_path  = "kowalski.png"
    image_kowalski = image.imread(im_path)
    call_kowalski()
    r_center = initialise_r_center()
    dr       = r_center[1] - r_center[0]
    mbar     = initialise_mbar()
    temp     = initialise_T()
    v        = initialise_v()
    P, rho, n, phi = initialise_wellbalanced(r_center, temp, mbar)

    w        = populate_primitive(rho, v, P)
    
    U, F, S  = primitive_to_conservative(w, phi, r_center)
    dphidr   = calc_dphi_dr(phi, r_center)
    
    # FIXED: Unpack the tuple correctly[cite: 1]
    Q_tot, Q_cools, Q_heats = compute_heating_cooling(w, r_center)

    check_well_balanced_residual(r_center, P, rho, phi, temp, mbar)

    plt.ion()
    # FIXED: Changed subplots to (1, 5) to match the unpacked axes[cite: 1]
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle("Kowalski — live simulation", fontsize=13)
    ax_rho, ax_v, ax_T, ax_s, ax_hc = axes

    P_bar = w[p.iP] / c.BAR_TO_CGS
    T_now = w[p.iP] * p.m_bar / (w[p.irho] * c.K_B_cgs)   # ideal gas: T = P*mbar / rho*kB

    def init_panel(ax, xdata, xlabel, color, xlog=False):
        line, = ax.plot(xdata, P_bar, lw=2, c=color)
        ax.set_ylabel("Pressure [bar]")
        ax.set_xlabel(xlabel)
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylim(1e-3, 1e-10)
        if xlog:
            ax.set_xscale('log')
        return line
    
    ax_v.set_xscale('log')
    ax_v.set_xlim(1e-6, 1e4)
    ax_T.set_xlim(0,6000)
    im_kowalski = ax_s.imshow(image_kowalski, aspect='auto', extent=(0.4, 0.6, .5, .7))
    ax_s.get_yaxis().set_visible(False)
    ax_s.get_xaxis().set_visible(False)
    ax_s.set_title("Stability analysis")
    line_rho = init_panel(ax_rho, w[p.irho],     "Density [g cm⁻³]",  "darkorange", xlog=True)
    line_v   = init_panel(ax_v,   w[p.iv] / 1e5, "abs Velocity [km s⁻¹]", "darkgreen")
    line_T   = init_panel(ax_T,   T_now,        "Temperature [K]",   "crimson")
    
    # Initialize the Heating & Cooling Panel
    ax_hc.set_ylabel("Pressure [bar]")
    ax_hc.set_xlabel("Rates [erg s⁻¹ cm⁻³]")
    ax_hc.set_yscale('log')
    ax_hc.set_xscale('log') # Log scale is best for plotting magnitude of rates
    ax_hc.invert_yaxis()
    ax_hc.set_ylim(1e-3, 1e-10)
    ax_hc.set_xlim(1e-30, 1e-3)
    
    # Define labels mapping to the Q_cools and Q_heats lists from compute_heating_cooling
    labels_cool = ["Mg I", "Mg II", "Na II", "K I", "Free-Free", "CH4", "CO"]
    labels_heat = ["XUV"]
    
    # Create line objects for each cooling and heating effect using absolute values
    lines_cool = [ax_hc.plot(np.abs(q) + 1e-30, P_bar, ls='--', lw=1.5, label=lbl)[0] for q, lbl in zip(Q_cools, labels_cool)]
    lines_heat = [ax_hc.plot(np.abs(q) + 1e-30, P_bar, ls='-', lw=2.0, label=lbl)[0] for q, lbl in zip(Q_heats, labels_heat)]
    ax_hc.set_xlim(1e-30, 1e-3)
    ax_hc.legend(loc='upper right', fontsize='x-small')

    time_text = fig.text(0.5, 0.01, "t = 0.00 s", ha="center", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.pause(0.01)

    t    = 0.0
    step = 0
    old_T = np.zeros(p.n_cells)
    max_dt = 0
    dphidr_init = calc_dphi_dr(phi, r_center)
    P_grad_init = (w[p.iP, 2:] - w[p.iP, :-2]) / (2 * dr)
    grav_init   = w[p.irho, 1:-1] * dphidr_init[1:-1]
    res_init    = np.max(np.abs(P_grad_init + grav_init))
    print(f"Residual at t=0: {res_init:.2e}")

    while t < t_max:
            
        U      = apply_boundary_conditions(U, phi, r_center)
        w      = conservative_to_primitive(U, r_center)
        dphidr = calc_dphi_dr(phi, r_center)
        
        # FIXED: Correctly unpack the tuple so update_S gets the total scalar[cite: 1]
        Q_tot, Q_cools, Q_heats = compute_heating_cooling(w, r_center)  
        S      = update_S(w, dphidr, Q_tot)
        
        dt     = calc_dt(r_center, w)
        fluxes = calc_face_fluxes(U, w, phi, r_center)
        U      = update_U(U, fluxes, S, dt, dr)

        t    += dt
        step += 1

        P_grad       = (w[p.iP, 2:] - w[p.iP, :-2]) / (2 * dr)
        grav_force   = w[p.irho, 1:-1] * dphidr[1:-1]
        max_residual = np.max(np.abs(P_grad + grav_force))
        
        if step == 1e9:
            pert_start = 200
            pert_end = 400
            print("Adding density perturbation")
            w[p.irho, pert_start:pert_end] = w[p.irho, pert_start:pert_end] * 2
            # Recompute U consistently from the perturbed w
            E_perturbed = w[p.iP,pert_start:pert_end] / (p.gamma - 1) + 0.5 * w[p.irho, pert_start:pert_end] * w[p.iv, pert_start:pert_end]**2
            U[p.irho, pert_start:pert_end] = w[p.irho, pert_start:pert_end]
            U[p.im,   pert_start:pert_end] = w[p.irho, pert_start:pert_end] * w[p.iv, pert_start:pert_end]
            U[p.ie,   pert_start:pert_end] = E_perturbed
            
        if step % plot_every == 0:
            
            P_now = w[p.iP] / c.BAR_TO_CGS
            T_now = w[p.iP] * p.m_bar / (w[p.irho] * c.K_B_cgs)

            line_rho.set_xdata(w[p.irho])
            line_v.set_xdata(abs(w[p.iv]) / 1e5)
            line_T.set_xdata(T_now)
            
            # NEW: Update the x-data for all heating and cooling lines
            for line, q in zip(lines_cool, Q_cools):
                line.set_xdata(np.abs(q) + 1e-30)
                line.set_ydata(P_now)
                
            for line, q in zip(lines_heat, Q_heats):
                line.set_xdata(np.abs(q) + 1e-30)
                line.set_ydata(P_now)
            ax_hc.set_xlim(1e-30, 1e-3)
            kowalski = modify_kowalski(image_kowalski, U, w, phi, r_center)
            im_kowalski.set_data(kowalski)
            
            for line in [line_rho, line_v, line_T]:
                line.set_ydata(P_now)

            for ax in axes:
                ax.autoscale_view()
                # NEW: Reset limits for ax_hc dynamically based on rate magnitudes
                if ax == ax_hc:
                    min_x = max(1e-30, np.min([np.min(np.abs(q) + 1e-30) for q in Q_cools + Q_heats]))
                    max_x = max(1e-10, np.max([np.max(np.abs(q)) for q in Q_cools + Q_heats]))
                    ax.set_xlim(min_x / 10, max_x * 10)

            max_dt = (max(abs(T_now - old_T)))
            time_text.set_text(
                f"t = {t:.4e} s | step = {step} | max residual = {max_residual:.2e}, | max dt = {max_dt:.2e}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            old_T = T_now

    plt.ioff()
    plt.show()
    print(f"Finished: {step} steps, t = {t:.4e} s")
    
analysis(t_max=1e4)   # run for 10,000 seconds



#####
"""
TODO
Add timestepping according to RKF2 (2.2 Käppeli 2016)
Look at pressure reconstruction 2.1.3
"""
