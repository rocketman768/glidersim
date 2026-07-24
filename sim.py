#!/usr/bin/env python3
"""
2DOF Point-Mass Glider Thermal Trajectory & Energy Model

Simulates longitudinal flight dynamics and total specific energy height (E_h)
evolution for an 18m glider traversing a spatial updraft (thermal).

Key Physical & Dynamic Features:
  - Integrated State: [x, z, v, gamma, n] resolved using 4th-order Runge-Kutta (RK4).
  - Updraft Profile: Continuous Allen thermal model with finite core and sink ring.
  - Shear Coupling: Coupled spatial gradient (dw/dx) terms in v_dot and gamma_dot
    to capture momentum and pitch-rate interactions in vertical wind gradients.
  - Actuator Dynamics: First-order low-pass lag (tau_n) modeling variometer, pilot,
    and airframe load factor response delays.
  - Control Law: Closed-loop PD speed-to-load-factor controller with n-clamping.
  - Drag Model: Empirical profile drag fit (Cd0) + Oswald induced drag (Cdi).

Output:
  Prints horizontal distance (x) versus Total Specific Energy Height (z + v^2 / 2g)
  to evaluate energy extraction efficiency across varied pull-up profiles.
"""

import math
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
import simCheater

# Physical constants
rho = 1.22 # kg/m^2
g = 9.82 # m/s^2

# Aircraft constants
eOswald = 0.95
S = 10.0 # m^2 (wing area)
m = 500.0 # kg (mass)
b = 18.0 # m (span)
AR = b * b / S # (aspect ratio)
tau_n = 0.5 # s (lag time constant for changes in load factor)

# State vars
state_t = None # s
state_x = None # m
state_z = None # m
state_v = None # m/s
state_gamma = None # radians (note: initialize this to proper steady-state for x/z/v)
state_nCmd = 1.0 # Commanded Gs of acceleration
state_n = 1.0 # Gs of acceleration. Control input.

# Simulation constants
dt = 0.1 # s
targetCruise_v = 49.0 # m/s
thermalWidth = 300.0 # m
thermalVelocity = 2.5 # m/s
thermalCenter = 450.0 # m

# Pilot profiles
@dataclass
class PilotProfile:
    name: str
    kp: float      # Speed error gain (g's per m/s speed error)
    kd: float      # Acceleration damping gain (g's per m/s^2 acceleration)
    targetDolphin_v: float # Target dolphin speed (m/s)
    n_max: float   # Upper load factor limit (g)
    n_min: float   # Lower load factor limit (g)
    v_pullThresh: float # Pilot pulls once the thermal is stronger than this (m/s)
    v_pushThresh: float # Pilot pushes once the thermal becomes weaker than this (m/s)
    x_cheaterPull: float = None
    x_cheaterPush: float = None
    nCmd_cheater: list[float] = None
    nCmd_x_cheater: list[float] = None

PILOT_BLOCK = PilotProfile(
    name="Block STF",
    kp=0.10,
    kd=0.38,
    targetDolphin_v=targetCruise_v,
    n_max=1.2,
    n_min=0.8,
    v_pullThresh=1.5,
    v_pushThresh=1.5,
)

PILOT_SMOOTH = PilotProfile(
    name="SmoothOperator",
    kp=0.10404040404040404,
    kd=0.7757575757575759,
    targetDolphin_v=25.0,
    n_max=1.2,
    n_min=0.8,
    v_pullThresh=0.0,
    v_pushThresh=0.9848484848484849,
)

PILOT_AGGRESSIVE = PilotProfile(
    name="AggroCraig",
    kp=0.10404040404040404,
    kd=0.5252525252525253,
    targetDolphin_v=29.363636363636363,
    n_max=2.0,
    n_min=0.5,
    v_pullThresh=1.5404040404040402,
    v_pushThresh=2.095959595959596,
)

PILOT_CHEATER = PilotProfile(
    name="Cheater",
    kp=0,
    kd=0,
    targetDolphin_v=25,
    n_max=3.0,
    n_min=0,
    v_pullThresh=0,
    v_pushThresh=0,
    nCmd_x_cheater=simCheater.initialCheaterParameters(thermalCenter, thermalWidth)[0],
    nCmd_cheater=simCheater.initialCheaterParameters(thermalCenter, thermalWidth)[1],
)

PILOT_OPTIMIZED = PilotProfile(
    name="Maverick",
    kp=0.06565656565656565,
    kd=0.29090909090909095,
    targetDolphin_v=25.0,
    n_max=3.0,
    n_min=0.0,
    v_pullThresh=1.5151515151515151,
    v_pushThresh=1.5656565656565657,
)

# Pick the pilot!
pilot = PILOT_CHEATER

class SimulationError(Exception):
    """Raised when the simulation enters an unphysical or numerical singularity state."""
    pass

# Pitch angle for given speed and loading
def steadyStateGamma(v, n_cmd):
    cl = commandedLiftCoefficient(n_cmd, v)
    cd = cd0(cl) + cdi(cl)
    drag = q(v) * S * cd
    return math.asin(-drag / (m * g))

def initializeState():
    global state_t, state_x, state_z, state_v, state_gamma, state_n
    state_t = 0.0 # s
    state_x = 0.0 # m
    state_z = 0.0 # m
    state_v = targetCruise_v # m/s
    state_n = 1.0 # m/s
    state_gamma = steadyStateGamma(state_v, state_n)

def w_box(x):
    if x < thermalCenter - thermalWidth/2 or x > thermalCenter + thermalWidth/2:
        return 0.0
    else:
        return thermalVelocity

def w_allen(x):
    x_c = thermalCenter
    r0 = thermalWidth / 3        # Radius of zero-lift crossover
    w_peak = thermalVelocity     # Peak core lift (m/s)
    
    r = abs(x - x_c)
    norm_r = r / r0
    
    # Allen formula
    return w_peak * (1.0 - norm_r**2) * math.exp(-(norm_r**2))

def dw_dx_allen(x):
    x_c = thermalCenter
    r0 = thermalWidth / 3        # Radius of zero-lift crossover
    w_peak = thermalVelocity     # Peak core lift (m/s)

    u = (x - x_c) / r0

    u_sq = u * u
    return (w_peak / r0) * 2.0 * u * (u_sq - 2.0) * math.exp(-u_sq)

# Air motion model in vertical m/s
def w(x):
    return w_allen(x)

def dw_dx(x):
    """Central difference derivative of w(x)."""
    #dx_step = 0.01
    #return (w(x + dx_step) - w(x - dx_step)) / (2.0 * dx_step)
    return dw_dx_allen(x)

def q(v):
    return 0.5 * rho * v * v

def commandedLiftCoefficient(n0, v):
    cl = n0 * m * g / (q(v) * S)
    return cl

def lift(v, cl):
    return q(v) * S * cl

# Valid for cl in [0.281, 1.33]
def cd0(cl):
    # 3 points from polar
    # (0.281, 36.7e-4)
    # (0.734, 45.0e-4)
    # (1.33, 70e-4)

    # Horner form quadratic
    u = cl * 2.252e-3 - 4.54e-4
    v = u * cl + 3.620e-3

    # The polar ignores drag from the fuselage and tail.
    # Add some to make it more realistic
    v += 30e-4

    # End of the drag buckets...approximate the nixus polar dropoffs
    if cl < 0.281:
        v += (0.281 - cl) * 50e-4/0.281
    elif cl > 1.33:
        v += (cl - 1.33) * 100e-4/0.1

    return v

def cdi(cl):
    return cl * cl / (math.pi * AR * eOswald)

def sinkRateInStillAir(v):
    """Calculates steady-state unaccelerated sink rate (m/s) at airspeed v."""
    cl = commandedLiftCoefficient(1.0, v)
    cd = cd0(cl) + cdi(cl)
    drag = q(v) * S * cd
    return (drag * v) / (m * g)

def impliedMacCready(v_cruise):
    """Computes the implied MacCready climb rate w_mc (m/s) for a given cruise speed."""
    dv = 0.01
    w_sink = sinkRateInStillAir(v_cruise)
    dw_dv = (sinkRateInStillAir(v_cruise + dv) - sinkRateInStillAir(v_cruise - dv)) / (2.0 * dv)
    
    # MacCready tangent intercept: w_mc = v * (dw/dv) - w_sink
    return v_cruise * dw_dv - w_sink

def maccreadyDolphinSpeed(x, v_cruise):
    """
    Computes the quasi-steady MacCready speed-to-fly at position x and the 
    spatial speed derivative (dv/dx) required to track the MacCready schedule.

    Returns:
        v_stf: Optimal instantaneous speed-to-fly (m/s)
        dv_dx: Target spatial acceleration dv/dx (1/s)
    """
    w_mc = impliedMacCready(v_cruise)
    w_val = w(x)
    dw_dx_val = dw_dx(x)

    # MacCready equation is only valid for w(x) < w_mc.
    # When w(x) >= w_mc, optimal straight speed is V_min_sink.
    V_MIN_SINK=25.0
    if w_val >= w_mc * 0.99:
        return V_MIN_SINK, 0.0
    
    # MacCready target intercept: f(v) = v * w_s'(v) - w_s(v) = w_mc - w(x)
    target = w_mc - w_val
    
    # Solve for v_stf via Newton-Raphson
    v = v_cruise  # Initial guess
    dv = 0.01     # Finite difference step for numerical derivatives

    # Newton-Raphson
    for _ in range(10):
        w_sink = sinkRateInStillAir(v)
        w_sink_plus = sinkRateInStillAir(v + dv)
        w_sink_minus = sinkRateInStillAir(v - dv)
        
        # Numerical 1st and 2nd derivatives of polar sink rate w_s(v)
        dw_dv = (w_sink_plus - w_sink_minus) / (2.0 * dv)
        d2w_dv2 = (w_sink_plus - 2.0 * w_sink + w_sink_minus) / (dv ** 2)
        
        f = v * dw_dv - w_sink - target
        f_prime = v * d2w_dv2  # df/dv
        
        if abs(f_prime) < 1e-9:
            break
            
        step = f / f_prime
        v -= step
        
        if abs(step) < 1e-5:
            break

    # Re-evaluate polar curvature at final v
    w_sink = sinkRateInStillAir(v)
    d2w_dv2 = (sinkRateInStillAir(v + dv) - 2.0 * w_sink + sinkRateInStillAir(v - dv)) / (dv ** 2)
    
    # Required spatial speed gradient: dv/dx = -1 / (v * w_s''(v)) * (dw/dx)
    denom = v * d2w_dv2
    dv_dx = -dw_dx_val / denom if denom > 1e-6 else 0.0

    return v, dv_dx

def derivatives(x, z, v, gamma, n, n_cmd):
    """Calculates [dx/dt, dz/dt, dv/dt, dgamma/dt, dn/dt] for a given state."""
    cl = commandedLiftCoefficient(n, v)
    cd = cd0(cl) + cdi(cl)
    drag = q(v) * S * cd

    # Wind shear spatial gradient
    shear = dw_dx(x)
    
    dx_dt = v * math.cos(gamma)
    dz_dt = v * math.sin(gamma) + w(x)

    # Coupled state derivatives
    dv_dt = -(drag / m) - g * math.sin(gamma) - v * shear * math.cos(gamma) * math.sin(gamma)
    dgamma_dt = (g / v) * (n - math.cos(gamma)) - shear * (math.cos(gamma) ** 2)
    
    # Dynamic lag derivative
    dn_dt = (n_cmd - n) / tau_n

    return dx_dt, dz_dt, dv_dt, dgamma_dt, dn_dt

def controlUpdate_cheater():
    x = pilot.nCmd_x_cheater
    n = pilot.nCmd_cheater

    def findNdx():
        for ndx, x0 in enumerate(x):
            if x0 > state_x:
                return max(0, ndx - 1)
        return len(x) - 1

    ndx = findNdx()
    if ndx < 0:
        return n[0]
    if ndx >= len(x) - 1:
        return n[-1]

    # linear n for each dx segment
    xleft = x[ndx]
    xright = x[ndx+1]
    a = (state_x - xleft) / (xright - xleft)
    nleft = n[ndx]
    nright = n[ndx+1]

    return (1-a) * nleft + a * nright

def controlUpdate():
    if pilot.nCmd_cheater and pilot.nCmd_x_cheater:
        return controlUpdate_cheater()

    # 0.2 per 2 m/s
    kp = pilot.kp
    # Tune to prevent overshoot
    kd = pilot.kd
    n_max = pilot.n_max
    n_min = pilot.n_min
    v_pullThresh = max(0, pilot.v_pullThresh)
    v_pushThresh = max(0, pilot.v_pushThresh)
    targetDolphin_v = pilot.targetDolphin_v

    # We need the velocity derivative. (pass dummy 0.0 for n_cmd since dv/dt doesn't use it)
    _, _, v_dot, _, _ = derivatives(state_x, state_z, state_v, state_gamma, state_n, 0.0)

    localShear = dw_dx(state_x)
    localW = w(state_x)
    if localShear >= 0:
        # Either coming in to the core or exiting the sink on the far side
        if localW < v_pullThresh:
            target_v = targetCruise_v
        else:
            target_v = targetDolphin_v
    elif localShear < 0:
        # Either passing the core or entering the sink on the near side
        if localW < v_pushThresh:
            target_v = targetCruise_v
        else:
            target_v = targetDolphin_v

    # Proportional term   
    n_cmd = 1.0 + kp * (state_v - target_v)
    # derivative term
    n_cmd += kd * v_dot
    # Clamp
    return max(n_min, min(n_max, n_cmd))

def advanceState():
    # RK4 update
    global state_t, state_x, state_z, state_v, state_gamma, state_n, state_nCmd

    # We have a simulation singularity at state_v = 0 where cl becomes infinite.
    # We must stop early on such singularities
    cl = commandedLiftCoefficient(state_n, max(1e-3, state_v))
    if cl > 5.0:
        raise SimulationError("Glider is deeply stalled")

    # 1. Update control input based on current state
    state_nCmd = controlUpdate()
    
    # 2. RK4 Intermediate steps
    # k1
    k1_x, k1_z, k1_v, k1_g, k1_n = derivatives(state_x, state_z, state_v, state_gamma, state_n, state_nCmd)
    
    # k2
    k2_x, k2_z, k2_v, k2_g, k2_n = derivatives(
        state_x + 0.5 * dt * k1_x,
        state_z + 0.5 * dt * k1_z,
        state_v + 0.5 * dt * k1_v,
        state_gamma + 0.5 * dt * k1_g,
        state_n + 0.5 * dt * k1_n,
        state_nCmd
    )
    
    # k3
    k3_x, k3_z, k3_v, k3_g, k3_n = derivatives(
        state_x + 0.5 * dt * k2_x,
        state_z + 0.5 * dt * k2_z,
        state_v + 0.5 * dt * k2_v,
        state_gamma + 0.5 * dt * k2_g,
        state_n + 0.5 * dt * k2_n,
        state_nCmd
    )
    
    # k4
    k4_x, k4_z, k4_v, k4_g, k4_n = derivatives(
        state_x + dt * k3_x,
        state_z + dt * k3_z,
        state_v + dt * k3_v,
        state_gamma + dt * k3_g,
        state_n + dt * k3_n,
        state_nCmd
    )
    
    # 3. Weighted state updates
    state_x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    state_z += (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    state_v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    state_gamma += (dt / 6.0) * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g)
    state_n += (dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
    
    state_t += dt

def energyAsHeight():
    return state_z + state_v ** 2 / (2 * g)

def detrendedEnergyHeight():
    # Adjust the total energy height for maccready
    w_mc = impliedMacCready(targetCruise_v)
    totalEnergy_height = state_z + state_v ** 2 / (2 * g) - w_mc * state_t

    # Steady cruise slope (m of energy height lost per m of horizontal distance)
    s_eff = (sinkRateInStillAir(targetCruise_v) + w_mc) / targetCruise_v
    # Detrended energy height
    baseline_E_h = - (s_eff * state_x)

    delta_E_h = (totalEnergy_height - baseline_E_h)
    return delta_E_h

def printState():
    v_kt = state_v * 1.94
    x_ft = state_x * 3.28
    z_ft = state_z * 3.28
    pitch_deg = state_gamma / math.pi * 180.0
    print(f'{state_t:.1f}\t{v_kt:.1f}\t{x_ft:.0f}\t{z_ft:.1f}\t{pitch_deg:.1f}\t{state_n:.1f}')

    #delta_E_h = detrendedEnergyHeight()

    #delta_E_h_ft = delta_E_h * 3.28
    #print(f'{x_ft:.0f}\t{delta_E_h_ft:.0f}')

def simulate(tMax, simPilot=PILOT_SMOOTH):
    global pilot

    pilot = simPilot
    initializeState()

    history = {"t": [], 
               "x": [], 
               "z": [], 
               "v": [], 
               "gamma": [], 
               "n": [], 
               "n_cmd": [], 
               "E_h": [], 
               "E_h_detrended": []
               }
    history['simulationOK'] = True

    while state_t < tMax:
        history['t'].append(state_t)
        history['x'].append(state_x)
        history['z'].append(state_z)
        history['v'].append(state_v)
        history['gamma'].append(state_gamma)
        history['n'].append(state_n)
        history['n_cmd'].append(state_nCmd)
        history['E_h'].append(energyAsHeight())
        history['E_h_detrended'].append(detrendedEnergyHeight())
        try:
            advanceState()
        except SimulationError as e:
            print(f'Simulation error. Stopping early. Reason: {e}', file=sys.stderr)
            history['simulationOK'] = False
            raise
    
    return history

if __name__ == '__main__':

    def plot(ax, x, y, removeOffset=False, annotateLastPoint=False):
        # Remove the initial offset
        if removeOffset:
            y = [yy - y[0] for yy in y]

        # Plot x,y and update legend
        ax.plot(x, y, label=f'{p.name}')
        ax.legend(loc="lower right")

        # Annotate last point
        if annotateLastPoint:
            lastPoint = (x[-1], y[-1])
            lastPointLabel = f'{lastPoint[1]:.2f}'
            ax.annotate(
                text=lastPointLabel,
                xy=lastPoint,
                xytext=(10, 5),                # Offset the text by 10 points right, 5 points up
                textcoords='offset points',    # Tells matplotlib to interpret xytext as pixel/point offsets
                fontsize=10,
                fontweight='bold'
            )
    # interactive mode
    plt.ion()

    dataX = 'x'
    dataY = 'v'

    # Set up the figure and axis
    #fig, ax = plt.subplots(figsize=(8, 5))
    fig = plt.figure()

    dataKeysToPlot = [('x','E_h_detrended'), ('x', 'v'), ('x', 'n'), ('x', 'n_cmd'), ('x', 'cl'), ('x', 'cd')]
    numCols = 2
    numRows = int((len(dataKeysToPlot) + numCols - 1) / numCols)
    axes = []
    for (keyX, keyY) in dataKeysToPlot:
        ax = fig.add_subplot(numRows, numCols, len(axes) + 1)
        ax.set_xlabel(keyX)
        ax.set_ylabel(keyY)
        ax.grid(True)
        axes.append(ax)

    # Plot ideal maccready stf
    #maccready_x = range(0,1400,10)
    #maccready_v = [maccreadyDolphinSpeed(x, 49)[0] for x in maccready_x]
    #ax.plot(maccready_x, maccready_v, label='Maccready')
    #ax.legend(loc='lower right')

    pilots = (PILOT_BLOCK, PILOT_SMOOTH, PILOT_OPTIMIZED, PILOT_AGGRESSIVE, PILOT_CHEATER)

    for p in pilots:
        data = simulate(30.0, p)
        # Add some things to the data to visualize
        data['cl'] = [commandedLiftCoefficient(n, v) for n,v in zip(data['n'], data['v'])]
        data['cd'] = [cd0(cl) + cdi(cl) for cl in data['cl']]

        for (ax, (keyX, keyY)) in zip(axes, dataKeysToPlot):
            annotateLastPoint = keyX == 'E_h_detrended'
            plot(ax, data[keyX], data[keyY])
            if keyY == 'cl':
                ax.axhline(y=0.281, color='gray', linestyle='--')
        
        # Force Matplotlib to redraw the frame and pause
        plt.draw()

    # Keep the final window open when the loop finishes
    plt.ioff()
    plt.show()