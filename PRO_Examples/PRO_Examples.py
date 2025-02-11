import sys 
import os
#For finding PROpy if not in the base directory (which it is in the default repo, developed separately though)
#sys.path.append(os.path.abspath("<PATH TO PROpy>"))

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import pro_lib
import numpy as np




#Class constants (Earth parameters) can be overwritten locally or when calling the utility methods
mu = 398600.432896939164493230  #gravitational constant
r_e = 6378.136  # Earth Radius 
J2 = 0.001082627 #J2 Constant

def main():
    """
    Main method of the orbit analysis tool where the orbit to analyze is initialized
    
    """
    
    #Example initial positions of the deputies around the chief 
    #In LVLH coordinate frame, x,y,z, in km

    #
    ## 3 s/c example
    #np.array([[ -0.12207855,  20.75256913,   4.74625255],
    #          [ -1.70607441,  -0.89786518,  -4.52224124],
    #          [  1.45510593, -20.8936754 ,   4.25425026]])

    # 6 s/c example (note the chief is always at 0,0,0)

    initDeputy = np.array([[  0.52460662,   4.4225015 ,   6.95510887],
                           [ -7.39308199, -35.4905448 ,  -4.810119  ],
                           [-12.29013267,  33.65355964,  -3.95438927],
                           [ -5.29459157,  -1.78675105,  -5.5510786 ],
                           [ -9.39245339,   9.08063789,   4.9918793 ]])
    
    
    #Time we integrate each orbit over (this is from 0 to 12 days, every minute)
    time = np.arange(0,12*24*3600,60)
     
    # orbital parameters, wrapped in a dictionary

    #Run on nominal NISAR mission (a radar science mission), but this can be changed to any situation.
    #Sun Sync periodic, N = 173, D = 12 (slight errors, probably due to higher order terms than J2 or minor corrections to the publicly availible data)
    #alt = 747, i = 98.4, e = 0 (assumed), other params unknown

    
    orbParams = {"time":time, #time steps over which to evaluate
                "NoRev":173, # revolutions considered, orbit altitude, orbit 
                "altitude":747, #orbit altitude
                "ecc":0, #orbit eccentricity
                "inc":98.4, #orbit inclination (deg)
                "Om":0, #orbit right ascension of ascending node (deg)
                "om":0, #orbit argument of periapsis (deg)
                "f":0, #orbit true anomaly (deg)
                "num_deputy":len(initDeputy), 
                "mu":mu,  #gravitational parameter of Earth
                "r_e":r_e,  # Earth Radius 
                "J2":J2} #J2 Constant of Earth



    
    #Demo cost computation and visualization
    demo(initDeputy,orbParams)



def demo(initDeputy,orbParams,animateFlag=False,animationName = "animation.mp4"):
    """
    Function that demonstrates the integration of a
    sample trajectory. The default orbit used is based off the
    NISAR mission
    """

    print("start")
    orbitState  = computeOrbitDynamics(initDeputy,orbParams)

    #Plot the computed dynamics (set optional parameters to configure for animation)
    animationTools(orbitState, orbParams['time'],animateFlag=animateFlag,animationName=animationName)


def computeOrbitDynamics(state,orbParams):
    """
    Method to compute the orbit dynamics of from an initial formation 
    position

    Parameters
    ----------
    state : array, shape(numDeputies,3)
        initial deputy spatial configurations of the
        swarm. Should be (x,y,z) of each deputy in order
    orbParams : dict
        Dictionary of the orbital parameters used to compute
        the orbit. Required values are:
            time, NoRev, altitude, ecc, inc, Om, om, f, 
            num_deputy, mu, r_e, J2

            These are: time steps over which to evaluate, 
            revolutions considered, orbit altitude, orbit 
            eccentricity, orbit inclination (deg), orbit 
            right ascension of ascending node (deg), orbit
            argument of periapsis (deg), orbit true anomaly 
            (deg), number of deputies, 
            earth gravitational parameter, radius of earth,
            earth J2

    Returns
    ---------
    orbitState : array shape(len(time),6*(num_deputy+1))
        State vector of the orbit at each time specified.
        First 6 states are the chief orbital parameters.
                                            #Notation in Morgan et. all
        ys[0] = r       #Semi major axis    r0
        ys[1] = v_x     #Radial Velocity    vx0 
        ys[2] = h       #Angular momentum   h0 
        ys[3] = Omega   #RA of Ascending N. Omega0                    
        ys[4] = inc     #Inclination        inc0            
        ys[5] = theta   #True Anomaly       theta0      
        Each subsequent 6 states are a deputy's relative
        state in LVLH as (x,y,z,vx,vy,vz)
    """


    #compute initial conditions for the chief and deputy
    ys = pro_lib.initial_conditions_deputy("nonlinear_correction_linearized_j2_invariant",
                                            [orbParams["NoRev"],orbParams["altitude"],orbParams["ecc"],orbParams["inc"],orbParams["Om"],orbParams["om"],orbParams["f"],orbParams["num_deputy"]],
                                            state,orbParams["mu"],orbParams["r_e"],orbParams["J2"])
    
    #Integrate the relative dynamics and chief orbital elements using pro_lib's dynamics function
    orbitState  = odeint(pro_lib.dyn_chief_deputies,ys,orbParams["time"],args=(orbParams["mu"],orbParams["r_e"],orbParams["J2"],orbParams["num_deputy"]))
    return orbitState


def orbitPeriodComputation(orbParams,timeStepsPerOrbit):
    """
    This code computes the time for a full orbit, accounting 
    for J2 perturbations

    Parameters
    ----------
    orbParams : dict
        Dictionary of the orbital parameters used to compute
        the orbit. Required values are:
            NoRev, altitude, ecc, inc,
            num_deputy, mu, r_e, J2

            These are: 
            revolutions considered, orbit altitude, orbit 
            eccentricity, orbit inclination (deg),
            number of deputies, 
            earth gravitational parameter, radius of earth,
            earth J2
    timeStepsPerOrbit : int
        Number of time steps to be computed per orbit, which
        determines the spacing of the time steps in the 
        returned time vector

    Returns
    -------
    time : array, shape(NoRev*timeStepsPerOrbit)
        Time steps along the orbit. Last element is the 
        computed timespan

    """

    # Energy Matched 
    # J2 disturbance 
    # No drag

    k_J2 = (3/2)*orbParams["J2"]*orbParams["mu"]*(orbParams["r_e"]**2)

    # Orbital Elements
    a = orbParams["r_e"] + orbParams["altitude"]           # semimajor axis [km] (Assumed circular orbit)
    inc = orbParams["inc"]*np.pi/180             # inclination [rad]


    # Xu Wang Parameters
    h = np.sqrt(a*(1 - orbParams["ecc"]**2)*orbParams["mu"])           # angular momentum [km**2/s]


    # effective period of the orbit
    a_bar = a*(1 + 3*orbParams["J2"]*orbParams["r_e"]**2*(1-orbParams["ecc"]**2)**0.5/(4*h**4/orbParams["mu"]**2)*(3*np.cos(inc)**2 - 1))**(-2/3)  
    period = 2*np.pi/np.sqrt(orbParams["mu"])*a_bar**(3/2)  
 

    # simulation time
    time = np.linspace(0,orbParams["NoRev"]*period,int(period/(timeStepsPerOrbit)))  
    
    return time               # time vector with units of orbits instead of seconds

    


def animationTools(orbitState, time,azim=-100, elev=43, animateFlag=False,frames=None,animationName="animation.mp4",sliders=False):
    """
    Helper method to animate or provide lightweight 
    visualization of the formation dynamics. Several
    optional parameters configure the type of visualization
    or animation displayed

    Parameters
    ----------
    orbitState : array, shape(len(time),6*(num_deputy+1)
        State vector of the orbit at each time specified.
        First 6 states are the chief orbital parameters.
        Each subsequent 6 states are a deputy's relative
        state in LVLH as (x,y,z,vx,vy,vz) 
    time : array
        Time at which each state is provided
    azim : double, (default=-100)
        Azimuth angle of the initial plot rendering
    elev : double, (default=43)
        Elevation angle of the initial plot rendering
    animate : Boolean, (default=False)
        Flag to animate the formation over the orbit
    frames : int, (default=None)
        If animating, how many frames to animate. 
        Default of none animates full orbit
    animationName : string, (default="animation.mp4")
        If animating, name of output file (in local
        directory). NOTE: No overwrite protection!
    sliders : boolean, (default=False)
        Flag to produce plot with interactive sliders
        for the formation over its orbit history.
    """
    

    #Plot the relative orbit tracks, at a provided or arbitrary view angle (found to work well for these visualizations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("6 space craft formation in NISAR J2 dynamic orbit, LVLH frame")
    ax.set_xlabel("x, radial out from Earth (km)")
    ax.set_ylabel("y, along track (km)")
    ax.set_zlabel("z, cross track (km)")
    #ax.set_xlim(-500, 500)
    #ax.set_ylim(-500, 500)
    #ax.set_zlim(-500, 500)
    ax.azim = azim
    ax.elev = elev


    #Loop through each deputy
    for i in range(int(len(orbitState[0])/6-1)):
    
        ax.plot(orbitState[:,6*(i+1)],orbitState[:,6*(i+1)+1],orbitState[:,6*(i+1)+2])
    
    ax.plot([0],[0],[0],"ko")
    #Get sense of scale by finding largest distance from chief
    scale = np.max(orbitState[:,6:])
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    
    plt.show()
    
    if animateFlag or sliders:
        #Save the user selected "best" veiw for animation
        azimuth = ax.azim
        elevation = ax.elev
    
    #Show the orbit controlled by sliders (sort of works) if desired, so the user can manipulate the dynamics
    if sliders:
        fig = go.Figure()
        # Add traces, one for each slider step
        for state in orbitState:
            xs = state[6::6]
            ys = state[7::6]
            zs = state[8::6]
            fig.add_trace(
                go.Scatter3d(
                    visible=False,
                    x=xs,y=ys,z=zs),
                    range_x=[-1,1], range_y=[-1,1], range_z=[-1,1])
        
        # Make 0th trace visible
        fig.data[0].visible = True
        
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)
        
        sliders = [dict(
            active=0,
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders
        )
        
        
        fig.show()
    
    #Animate if desired
    if animateFlag:
        #Check if user specified number of frames, or animate whole thing
        if frames is None:
            frames = len(time)


        #Only import when animating
        import matplotlib.animation as animation
        orbitData = np.array([orbitState[:,6::6],orbitState[:,7::6],orbitState[:,8::6]])
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        
        fig = plt.figure(figsize=(10,10))
        fig.suptitle("Space craft formation in NISAR J2 dynamic orbit, LVLH frame", fontsize=14)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.azim = azimuth
        ax.elev = elevation
        ani = animation.FuncAnimation(fig, animate, frames=frames, fargs=(orbitData,ax,scale))
        
        ani.save(animationName, writer=writer)


def animate(i,orbitData,ax,scale):
    """
    Function that makes an animation over the course of an orbit

    Parameters
    ----------
    i : integer
        Minute along the orbit to animate
    orbitData : array, shape(3,len(time),num_deputy
        State vector of the x,y,z positions of the deputies 
        at each time along the orbit in the LVLH frame. First
        dimension is the x, y, or z direction, next dimension 
        is time, third dimension is the deputy.
    ax : matplotlib.pyplot.axis
        The axis we are animating on
    scale : double
        The maximum spatial extent of the formation in any of 
        the axis directions
    """
    print(i)
    ax.clear()
    ax.set_title("Time = " + str(i) + " minutes")
    ax.set_xlabel("x, radial out from Earth (km)")
    ax.set_ylabel("y, along track (km)")
    ax.set_zlabel("z, cross track (km)")
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    data = orbitData[:,int(i):int(i+1),:] #select data range
    for j in range(len(data[0,0,:])):
        ax.plot(data[0,:,j],data[1,:,j],data[2,:,j],"o")
    
    ax.plot([0],[0],[0],"ko")



if __name__ == "__main__":
    main()

   
