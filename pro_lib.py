#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math

"""
-------------------------------------------------------------------------
--------------------Library for the Swarm Relative Orbital Dynamics ------------------
-------------------------------------------------------------------------
Author: <Yashwanth Nakka>
Maintainer 1:<Yashwanth Nakka>Email:<ynakka@caltech.edu>
Version : 0.3

List of Functions:
    
    1) Pointing vector
    
    2) HCW (x,y,z) w.r.t chief

    3) HCW Radial Parameterization
    
    4) Chief and deputies stacked dynamics

    5) J2 Invariant, Energy Matched

    6) Density Model, along with derivative

    7) lvlh_to_eci transformations

    8) eci_to_lvlh transformations 

    9) initial_conditions 
    
    # notation consistent with the book and Morgan et. all (https://arc.aiaa.org/doi/10.2514/1.55705)
    # Spacecraft Formation Flying Dynamics, control and navigation
"""

def pointing_vector(sat_state,chief_state):
    """
    pointing vector expressed in local LVLH frame w.r.t the sat_frame
    """
    sat_position = np.zeros([3,1])
    for i in range(0,3):
        sat_position[i] = sat_state[i]

    chief_position = np.zeros([3,1])
    for i in range(0,3):
        chief_position[i] = chief_state[i]

    pt_vec = chief_position - sat_position

    return pt_vec


def pro_sample_hcw(state_init,mean_motion,tvec):

    num_sample = tvec.shape[0]
    state_vec = np.zeros((6,num_sample))

    for nn in range(num_sample):
        state_vec[:,nn] = hcw_state_space(state_init,mean_motion,tvec[nn])

    return state_vec


def hcw_state_space(state_init,mean_motion,t):
    """
    state_init = [x,y,z,dot{x},dot{y},dot{z}]
    Outputs state at any time 't'
    """
    xi = state_init[0]
    yi = state_init[1]
    zi = state_init[2]
    dxi= state_init[3]
    dyi= state_init[4]
    dzi= state_init[5]

    w = mean_motion

    x = (4*xi + (2*dyi)/w)+(dxi/w)*math.sin(w*t)-(3*xi+(2*dyi)/w)*math.cos(w*t)
    y = (yi - (2*dxi)/w)+((2*dxi)/w)*math.cos(w*t)+(6*xi + (4*dyi)/w)*math.sin(w*t)-(6*w*xi+3*dyi)*t
    z = zi*math.cos(w*t)+(dzi/w)*math.sin(w*t)
    
    dx = dxi*math.cos(w*t) + ((3*xi*w + 2*dyi)*math.sin(w*t))    
    dy = -(6*w*xi + 3*dyi) + ((6*xi*w + 4*dyi)*math.cos(w*t)) - (2*dxi*math.sin(w*t))
    dz = dzi*math.cos(w*t) - zi*w*math.sin(w*t)
    
    return([x,y,z,dx,dy,dz])

def hcw_phasemagnitude_to_statespace(magnitude,phase,mean_motion,t):
    rho_x = magnitude[0]
    rho_y = magnitude[1]
    rho_z = magnitude[2]
    
    alpha_x = phase[0]
    alpha_z = phase[1]

    x = rho_x* math.sin(mean_motion*t + alpha_x)
    y = rho_y + (2*rho_x*math.cos(mean_motion*t + alpha_x))
    z = rho_z*math.sin(mean_motion*t + alpha_z)
    
    dx = rho_x*mean_motion* math.cos(mean_motion*t + alpha_x)
    dy = -2*rho_x*mean_motion*math.sin(mean_motion*t + alpha_x)
    dz = rho_z*mean_motion*math.cos(mean_motion*t + alpha_z)
    
    # ToDo: return x_dot, y_dot, z_dot
    return ([x,y,z,dx,dy,dz])

def hcw_phasemagnitude(state_init,mean_motion):
    xi = state_init[0]
    yi = state_init[1]
    zi = state_init[2]
    dxi= state_init[3]
    dyi= state_init[4]
    dzi= state_init[5]

    n = mean_motion

    rho_x = math.sqrt(dxi**2 + ((xi**2)*(n**2)))/n
    rho_z = math.sqrt(dzi**2 + ((zi**2)*(n**2)))/n
    rho_y = yi - (2*dxi/n)

    alpha_x = math.atan2(n*xi,dxi)
    alpha_z = math.atan2(n*zi,dzi)

    magnitude = np.array([rho_x,rho_y,rho_z])
    phase = np.array([alpha_x,alpha_z])

    return magnitude, phase


def hcw_statespace_to_phasemagnitude(magnitude,phase,mean_motion):

    # this is not a one to one transformation 

    rho_x = magnitude[0]
    rho_y = magnitude[1]
    rho_z = magnitude[2]
    
    alpha_x = phase[0]
    alpha_z = phase[1]

    n = mean_motion
    tax = math.tan(alpha_x)
    taz = math.tan(alpha_z)

    dxi = math.sqrt(((n**2)*(rho_x**2))/(1 + tax**2))
    dzi = math.sqrt(((n**2)*(rho_z**2))/(1 + taz**2))

    xi = tax*dxi/n
    zi = taz*dzi/n
    yi = rho_y + (2*dxi/n)

    dyi = -2*n*xi

    state_init = np.array([xi,yi,zi,dxi,dyi,dzi])

    return state_init


def dyn_chief_deputies(y,t,mu=398600.4418,r_e=6378.1363,J2=1082.64*10**(-6),num_deputies=0):

    """
    Returns a vector containing the derivatives of each of the chief orbital 
    parameter and each element of states of the deputies, given the current 
    swarm state. Intended for use with scipy.integrate.odeint. Implements J2
    dynamics, neglects drag.
    
    Parameters
    ----------
    y : array, shape(6*(number of deputies+1))
        State of the swarm at this time instant. The first 6 entries are the
        chief orbit parameters, using Xu Wang Parameters, from Nonlinear 
        Dynamic Equations of Satellite Relative MotionAround an Oblate Earth 
        (https://arc.aiaa.org/doi/10.2514/1.33616).
        The next 6 should be the state of each deputy
    mu : double
        Gravitational constant (km^3/s^2). The default is 398600.4418 for Earth.
    r_e : double 
        Planet radius (km). The default is 6378.1363 for Earth.
    J2 : double
        J2 coefficient. The default is 1082.64*10**(-6) for Earth.
    num_deputies : integer
        Number of deputies to integrate. Default is none (should match y size TODO?)

    Returns
    -------
    dydt : array, shape(6*(number of deputies + 1))
        Derivatives of each element of the input y
    """

    # Chief Satellite states 
    r = y[0]
    vx = y[1]
    h = y[2]
    Omega = y[3]
    inc = y[4]
    theta = y[5]
    
    

    k_J2 = (3/2)*J2*mu*(r_e**2) # J2 "force" coefficient. For Earth it is 2.633*10**10, km**5/s**2

    dydt = np.zeros([int(6+6*num_deputies)])
    
    #Compute frequently used trig functions
    si = np.sin(inc)
    ci = np.cos(inc)
    st = np.sin(theta)
    ct = np.cos(theta)
    
    si2 = (np.sin(inc))**2
    ci2 = (np.cos(inc))**2
    st2 = (np.sin(theta))**2

    #Compute dynamics of chief hybrid orbital elements
    dydt[0] = vx
    dydt[1] = -mu/(r**2) + h**2/(r**3) - (k_J2/(r**4))*(1 - 3*si2*st2)
    dydt[2] = -(k_J2/(r**3))*(si2*np.sin(2*theta))
    dydt[3] = -(2*k_J2/(h*r**3))*(ci*st2)
    dydt[4] = -(k_J2/(2*h*r**3))*(np.sin(2*inc)*np.sin(2*theta) )
    dydt[5] = h/(r**2) + (2*k_J2/(h*r**3))*(ci2*st2)
    
    #Compute parameters used for deputy dynamics
    wx = -(k_J2/(h*r**3))*(np.sin(2*inc)*st)
    wz = h/r**2
    
    zeta = (2*k_J2/(r**4))*(si*st)
    
    eta_sq = mu/(r**3) + k_J2/(r**5) - ((5*k_J2/(r**5))*(si2*st2))
    
    alpha_z = -(2*h*vx/(r**3)) - (k_J2*si2*np.sin(2*theta)/(r**5))
    
    alpha_x = -(k_J2*np.sin(2*inc)*ct/(r**5)) + (3*vx*k_J2*np.sin(2*inc)*st/(h*(r**4))) - (8*(k_J2**2)*(si**3)*ci*(st**2)*ct)/((h**2)*(r**6))

    #Compute relative deputy dynamics under J2 effects
    for i in range(int(num_deputies)):
        xi = y[6*(i+1)+0]
        yi = y[6*(i+1)+1]
        zi = y[6*(i+1)+2]
        dxi = y[6*(i+1)+3]
        dyi = y[6*(i+1)+4]
        dzi = y[6*(i+1)+5]
        
        dydt[6*(i+1)+0] = dxi
        dydt[6*(i+1)+1] = dyi
        dydt[6*(i+1)+2] = dzi
        
        ri = np.sqrt((r+xi)**2 + yi**2 + zi**2)
        
        riZ = (r+xi)*si*st + yi*si*ct + zi*ci
        
        zetai = (2*k_J2*riZ/(ri**5))
        
        etai_sq = mu/(ri**3) + k_J2/(ri**5) - 5*k_J2*riZ**2/(ri**7)
        
        dydt[6*(i+1)+3] = 2*dyi*wz - xi*(etai_sq - wz**2) + yi*alpha_z - zi*wx*wz - (zetai - zeta)*si*st -r*(etai_sq-eta_sq)
        dydt[6*(i+1)+4] = -2*dxi*wz + 2*dzi*wx - xi*alpha_z - yi*(etai_sq - wz**2 - wx**2) + zi*alpha_x - (zetai - zeta)*si*ct
        dydt[6*(i+1)+5] = -2*dyi*wx -xi*wx*wz - yi*alpha_x - zi*(etai_sq - wx**2) - (zetai - zeta)*ci

    return dydt


def EulerInt(dxdt,dt,xt):
    xt1 = xt + dxdt*dt   
    return xt1



def airdensity(r,R_e):
    
    """
    upto r- Re = 1000 km
    Re = Radius of earth
    r = chief orbit radius
    """
    # R_e ---- Radius of Earth [km]
    rho = 0 
    h_0 = 0
    H   = 0

    h_ellp = r - R_e #altitute
    
    if h_ellp >= 0 and h_ellp < 25:
        h_0 = 0            # base altitude [km]
        rho_0 = 1.225       # nominal density [kg/m**3]
        H = 7.249             # scale height [km]
    elif h_ellp >= 25 and h_ellp < 30:
        h_0 = 25
        rho_0 = 3.899*10**(-2)
        H = 6.349
    elif (h_ellp >= 30) and (h_ellp < 40):
        h_0 = 30 
        rho_0 = 1.774*10**(-2) 
        H = 6.682 
    elif (h_ellp >= 40) and (h_ellp < 50):
        h_0 = 40 
        rho_0 = 3.972*10**(-3) 
        H = 7.554 
    elif (h_ellp >= 50) and (h_ellp < 60):
        h_0 = 50 
        rho_0 = 1.057*10**(-3) 
        H = 8.382 
    elif (h_ellp >= 60) and (h_ellp < 70):
        h_0 = 60 
        rho_0 = 3.206*10**(-4) 
        H = 7.714 
    elif (h_ellp >= 70) and (h_ellp < 80):
        h_0 = 70 
        rho_0 = 8.770*10**(-5) 
        H = 6.549 
    elif (h_ellp >= 80) and (h_ellp < 90):
        h_0 = 80 
        rho_0 = 1.905*10**(-5) 
        H = 5.799 
    elif (h_ellp >= 90) and (h_ellp < 100):
        h_0 = 90 
        rho_0 = 3.396*10**(-6) 
        H = 5.382 
    elif (h_ellp >= 100) and (h_ellp < 110):
        h_0 = 100 
        rho_0 = 5.297*10**(-7) 
        H = 5.877 
    elif (h_ellp >= 110) and (h_ellp < 120):
        h_0 = 110 
        rho_0 = 9.661*10**(-8) 
        H = 7.263 
    elif (h_ellp >= 120) and (h_ellp < 130):
        h_0 = 120 
        rho_0 = 2.438*10**(-8) 
        H = 9.473 
    elif (h_ellp >= 130) and (h_ellp < 140):
        h_0 = 130 
        rho_0 = 8.484*10**(-9) 
        H = 12.636 
    elif (h_ellp >= 140) and (h_ellp < 150):
        h_0 = 140 
        rho_0 = 3.845*10**(-9) 
        H = 16.149 
    elif (h_ellp >= 150) and (h_ellp < 180):
        h_0 = 150 
        rho_0 = 2.070*10**(-9) 
        H = 22.523 
    elif (h_ellp >= 180) and (h_ellp < 200):
        h_0 = 180 
        rho_0 = 5.464*10**(-10) 
        H = 29.740 
    elif (h_ellp >= 200) and (h_ellp < 250):
        h_0 = 200 
        rho_0 = 2.789*10**(-10) 
        H = 37.105 
    elif (h_ellp >= 250) and (h_ellp < 300):
        h_0 = 250 
        rho_0 = 7.248*10**(-11) 
        H = 45.546 
    elif (h_ellp >= 300) and (h_ellp < 350):
        h_0 = 300 
        rho_0 = 2.418*10**(-11) 
        H = 53.628 
    elif (h_ellp >= 350) and (h_ellp < 400):
        h_0 = 350 
        rho_0 = 9.518*10**(-12) 
        H = 53.298 
    elif (h_ellp >= 400) and (h_ellp < 450):
        h_0 = 400 
        rho_0 = 3.725*10**(-12) 
        H = 58.515 
    elif (h_ellp >= 450) and (h_ellp < 500):
        h_0 = 450 
        rho_0 = 1.585*10**(-12) 
        H = 60.828 
    elif (h_ellp >= 500) and (h_ellp < 600):
        h_0 = 500 
        rho_0 = 6.967*10**(-13) 
        H = 63.822 
    elif (h_ellp >= 600) and (h_ellp < 700):
        h_0 = 600 
        rho_0 = 1.454*10**(-13) 
        H = 71.835 
    elif (h_ellp >= 700) and (h_ellp < 800):
        h_0 = 700 
        rho_0 = 3.614*10**(-14) 
        H = 88.667 
    elif (h_ellp >= 800) and (h_ellp < 900):
        h_0 = 800 
        rho_0 = 1.170*10**(-14) 
        H = 126.64 
    elif (h_ellp >= 900) and (h_ellp < 1000):
        h_0 = 900 
        rho_0 = 5.245*10**(-15) 
        H = 181.05 
    else:
        h_0 = 1000 
        rho_0 = 3.019*10**(-15) 
        H = 268.00

    rho = rho_0*np.exp((h_0-h_ellp)/H)
    drhodr = rho_0*np.exp((h_0-h_ellp)/H)/H
    
    return rho, drhodr


def rotation_matrix_lvlh_to_eci(omega,theta,inc):

    """
    Uses Z-X-Z rotation 
    """
    # rotation about z 
    R3 = rotate_z(theta)

    # rotation about x
    R2 = rotate_x(inc)

    # rotation about z
    R1 = rotate_z(omega)

    R = np.array(np.mat(R1)*np.mat(R2)*np.mat(R3))

    return R

def rotation_matrix_eci_to_lvlh(omega,theta,inc):
    
    # rotation about z 
    R3 = rotate_z(theta)

    # rotation about x
    R2 = rotate_x(inc)

    # rotation about z
    R1 = rotate_z(omega)

    R = np.array(np.mat(R1)*np.mat(R2)*np.mat(R3))

    return R.T

def initial_conditions_deputy(initial_condition_type, input_info, initial_xyz, mu=398600.4418,r_e=6378.1363,J2=1082.64*10**(-6)):
    
    """
    Returns a state vector containing the chief orbital parameters and the
    initial state of the deputies for the desired swarm type by computing
    the required initial velocities.
    
    Parameters
    ----------
    initial_condition_type : string
        Specifies the type of swarm to initialize
    input_info : array
        Contents of the array are:
            input_info[0]: Number of orbits
            Chief Orbit parameters:
            input_info[1]: Orbit altitude in km
            input_info[2]: Orbit eccentricity
            input_info[3]: Orbit inclination (degrees)
            input_info[4]: Right Ascension of the Ascending Node (degrees)
            input_info[5]: Argument of Perigee (degrees)
            input_info[6]: True Anomaly (degrees)
            input_info[7]: Number of deputies
    initial_xyz : array, shape (number of deputies, 3)
        The initial LVLH position of each deputy relative to the chief. UNITS???
    mu : double
        Gravitational constant (km^3/s^2). The default is 398600.4418 for Earth.
    r_e : double 
        Planet radius (km). The default is 6378.1363 for Earth.
    J2 : double
        J2 coefficient. The default is 1082.64*10**(-6) for Earth.

    Returns
    -------
    ys : array, shape(6*(number of deputies + 1))
        First 6 elements are the hybrid orbital elements of the chief orbit
        ys[0:6] = (r0,vx0,h0,Omega0,inc0,theta0) 
                or (initial radius, initial radial velocity, initial angular momentum, 
                Right Ascension of the Ascending Node, initial inclination,
                initial True Anomaly)
        Each subsequent 6 elements are the state vector of the deputies in 
        the LVLH frame relative to the chief.

    Raises
    -------
    ValueError("initial_condition_type is not one of the valid options")
        If the provided initial_condition_type string is not one of the options:
        "uncontrolled_swarm","linearized_period_matched_swarm",
        "linearized_concentric_pro_swarm","no_crosstrack_drift_swarm",
         "linearized_j2_invariant_swarm","nonlinear_correction_linearized_j2_invariant"
    """

    # assigning parameters 
    NoRev = input_info[0] # number of orbits UNUSED
    # orbital parameters

    altitude = input_info[1]
    ecc = input_info[2]
    INC = input_info[3]
    Om = input_info[4]
    om = input_info[5]
    f = input_info[6]

    deputy_num = int(input_info[7])

    k_J2 = (3/2)*J2*mu*(r_e**2)     # J2 "force" coefficient. For Earth it is 2.633*10**10, km**5/s**2

    # Orbital Elements (converting to radians)
    a = r_e + altitude              # semimajor axis [km]
    inc = INC*np.pi/180             # inclination [rad]
    Omega = Om*np.pi/180            # RAAN [rad]
    omega = om*np.pi/180            # Arg of Per [rad]
    nu = f*np.pi/180                # True Anomaly [rad]

    # Xu Wang Parameters 
    # From Nonlinear Dynamic Equationsof Satellite Relative MotionAround an Oblate Earth 
    # (https://arc.aiaa.org/doi/10.2514/1.33616)
    h = np.sqrt(a*(1 - ecc**2)*mu)          # angular momentum [km**2/s]
    r = h**2/((1 + ecc*np.cos(nu))*mu)      # geocentric distance [km]
    v_x = mu*ecc*np.sin(nu)/h               # radial velocity [km/s]
    theta = omega + nu                      # argument of latitude [rad]

    # chief velocity
    v_c = np.sqrt(mu*(2/r - 1/a)) 

    # Number of Satellites
    sat_num = int(deputy_num + 1) 
    
    # Gradient of Gravitational Potential Energy (U)           
    gradU_x = mu/r**2 + k_J2/r**4*(1 - 3*np.sin(inc)**2*np.sin(theta)**2)   # x component of gradient of U
    gradU_y = k_J2*np.sin(inc)**2*np.sin(2*theta)/r**4                      # y component of gradient of U
    gradU_z = k_J2*np.sin(2*inc)*np.sin(theta)/r**4                         # z component of gradient of U
    gradU_xy = np.sqrt(gradU_x**2 + gradU_y**2)  
    gradU_mag = np.sqrt(gradU_xy**2 + gradU_z**2)                           # magnitude of gradient of U
    alpha = math.atan2(gradU_y,gradU_x)                                     # first rotation angle
    beta = math.atan2(gradU_z,gradU_xy)                                     # second rotation angle
    w = np.sqrt(gradU_mag/r)                                                # total rotation rate of LVLH frame
    w_x = -3/2*mu*J2*r_e**2*np.sin(2*inc)*np.sin(theta)/(h*r**3)            # precession rate of LVLH frame
    w_z = h/r**2                                                      

    # effective period of the orbit
    a_bar = a*(1 + 3*J2*r_e**2*(1-ecc**2)**0.5/(4*h**4/mu**2)*(3*np.cos(inc)**2 - 1))**(-2/3)  
    period = 2*np.pi/np.sqrt(mu)*a_bar**(3/2)  
    n = h/r**2                                                              # orbital rotation rate for non J2
    Pot_c = -mu/r - k_J2/(3*r**3)*(1 - 3*np.sin(inc)**2*np.sin(theta)**2)  


    tanth = math.tan(theta)
    if math.tan(theta) > 4:
        tanth =4
    
    # initial conditions for the chief (orbit parameters)
    r0 = r
    h0 = np.sqrt(r0*(1-ecc**2)*mu)
    vx0 = mu*ecc*np.sin(nu)/h0
    Omega0 = Omega
    inc0 = inc
    theta0 = omega + nu
    ys = np.zeros([sat_num*6],dtype = float)
                                        #Notation in Morgan et. all
    ys[0] = r       #Semi major axis    r0
    ys[1] = v_x     #Radial Velocity    vx0 
    ys[2] = h       #Angular momentum   h0 
    ys[3] = Omega   #RA of Ascending N. Omega0                    
    ys[4] = inc     #Inclination        inc0            
    ys[5] = theta   #True Anomaly       theta0                    
    
    
    # Initialize the initial conditions for the deputies in LVLH, 
    # as specified by the swarm dynamics requested

    # Uncontrolled Swarm, no relative velocity to chief, will drift
    if initial_condition_type == "uncontrolled_swarm":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = x
            ys[6*(i+1)+1] = y
            ys[6*(i+1)+2] = z
            ys[6*(i+1)+3] = 0
            ys[6*(i+1)+4] = 0
            ys[6*(i+1)+5] = 0
        return ys

    # Linearized period matching, solves HCW equations, will drift
    if initial_condition_type == "linearized_period_matched_swarm":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = x
            ys[6*(i+1)+1] = y
            ys[6*(i+1)+2] = z
            ys[6*(i+1)+3] = 0
            ys[6*(i+1)+4] = -2*n*x
            ys[6*(i+1)+5] = 0
        return ys
    
    # Linearized period matching. Initializes the swarm in concentric 
    # circular PROs to minimize collisions by solving HCW equations, 
    # will drift
    if initial_condition_type == "linearized_concentric_pro_swarm":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = initial_xyz[i,0]
            ys[6*(i+1)+1] = initial_xyz[i,1]
            ys[6*(i+1)+2] = initial_xyz[i,2]
            ys[6*(i+1)+3] = n*y/2
            ys[6*(i+1)+4] = -2*n*x
            ys[6*(i+1)+5] = 0
        return ys
    
    # An improvement on the previous case that adds periodic cross-track
    # motion that will not grow under J2, will drift.
    if initial_condition_type == "no_crosstrack_drift_swarm":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = x
            ys[6*(i+1)+1] = y
            ys[6*(i+1)+2] = z
            ys[6*(i+1)+3] = n*y/2
            ys[6*(i+1)+4] = -2*n*x
            ys[6*(i+1)+5] = -n*z*tanth
        return ys
    
    # Computes the needed initial velocity for a swarm that is invariant
    # under linear J2 conditions, will drift due to nonlinearities.
    if initial_condition_type == "linearized_j2_invariant_swarm":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = x
            ys[6*(i+1)+1] = y
            ys[6*(i+1)+2] = z

            # compute velocity initial conditions
            si = np.sin(inc)
            ci = np.cos(inc)
            st = np.sin(theta)
            ct = np.cos(theta)

            ca = np.cos(alpha)
            sa = np.sin(alpha)

            cb = np.cos(beta)
            sb = np.sin(beta)

            ca2 = np.cos(alpha)**2
            sa2 = np.sin(alpha)**2

            cb2 = np.cos(beta)**2
            sb2 = np.sin(beta)**2

            
            Pot_j = -(mu/np.sqrt((r+x)**2 + y**2 + z**2)) - (k_J2/(3*np.sqrt(((r+x)**2 + y**2 + z**2)**3)))*(1 - 3*((r + x)*si*st + y*si*ct + z*ci)**2/((r+x)**2 + y**2 + z**2))    
            
            dxi_j = w_z*(((3/2)*ca*sa*cb - ca2*sb2*tanth)*x + ((1/2)*ca2*cb + 2*sa2*cb - ca*sa*sb2*tanth)*y + (2*sa*sb + ca*cb*sb*tanth)*z) 
            dyi_j = w_z*((-(1/2)*sa2*cb - 2*ca2*cb - ca*sa*sb2*tanth)*x + (-(3/2)*ca*sa*cb -sa2*sb2*tanth)*y + (-2*ca*sb + sa*cb*sb*tanth)*z) 
            dzi_j = w_z*((-(1/2)*sa*sb + ca*cb*sb*tanth)*x + ((1/2)*ca*sb + sa*cb*sb*tanth)*y - (cb2*tanth)*z) 

            ys[6*(i+1)+3] = dxi_j 
            ys[6*(i+1)+4] = dyi_j
            ys[6*(i+1)+5] = dzi_j 
        return ys 

    # Computes the non linear initial conditions to be invariant under J2. 
    # Will drift at ~ 8mm/orbit under only J2 dynamics. 
    if initial_condition_type == "nonlinear_correction_linearized_j2_invariant":
        for i in range(deputy_num):
            x = initial_xyz[i,0]
            y = initial_xyz[i,1]
            z = initial_xyz[i,2]
            ys[6*i+6] = x
            ys[6*(i+1)+1] = y
            ys[6*(i+1)+2] = z

            # compute velocity initial conditions
            si = np.sin(inc)
            ci = np.cos(inc)
            st = np.sin(theta)
            ct = np.cos(theta)

            ca = np.cos(alpha)
            sa = np.sin(alpha)

            cb = np.cos(beta)
            sb = np.sin(beta)

            ca2 = np.cos(alpha)**2
            sa2 = np.sin(alpha)**2

            cb2 = np.cos(beta)**2
            sb2 = np.sin(beta)**2

            
            Pot_j = -(mu/np.sqrt((r+x)**2 + y**2 + z**2)) - (k_J2/(3*np.sqrt(((r+x)**2 + y**2 + z**2)**3)))*(1 - 3*((r + x)*si*st + y*si*ct + z*ci)**2/((r+x)**2 + y**2 + z**2))    
            
            dxi_j = w*(((3/2)*ca*sa*cb - ca2*sb2*tanth)*x + ((1/2)*ca2*cb + 2*sa2*cb - ca*sa*sb2*tanth)*y + (2*sa*sb + ca*cb*sb*tanth)*z) 
            dyi_j = w*((-(1/2)*sa2*cb - 2*ca2*cb - ca*sa*sb2*tanth)*x + (-(3/2)*ca*sa*cb -sa2*sb2*tanth)*y + (-2*ca*sb + sa*cb*sb*tanth)*z) 
            dzi_j = w*((-(1/2)*sa*sb + ca*cb*sb*tanth)*x + ((1/2)*ca*sb + sa*cb*sb*tanth)*y - (cb2*tanth)*z) 

            vxdi = v_x + dxi_j - n*y
            vydi = np.sqrt(v_c**2 - v_x**2) + dyi_j + n*x - w_x*z
            vzdi = dzi_j + w_x*y

            vdi_norm  = np.sqrt(vxdi**2 + vydi**2 + vzdi**2)  
            vrj_norm = np.sqrt(v_c**2 + 2*(Pot_c - Pot_j))

            vxj = vrj_norm*vxdi/vdi_norm
            vyj = vrj_norm*vydi/vdi_norm 
            vzj = vrj_norm*vzdi/vdi_norm    
            
            ys[6*(i+1)+3] = dxi_j -vxdi + vxj
            ys[6*(i+1)+4] = dyi_j -vydi + vyj
            ys[6*(i+1)+5] = dzi_j -vzdi + vzj
        return ys 
    else:
        raise ValueError("initial_condition_type is not one of the valid options")
        

def rotate_x(angle):
    R_x = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    return R_x

def rotate_y(angle):
    R_y = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    return R_y

def rotate_z(angle):
    R_z = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    return R_z

if __name__ == "__main__":
    print('This is a library of functions.')
