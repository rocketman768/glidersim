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
from simConstants import *
import simPilot

state = simPilot.SimState(None, None, None, None, None, None, 1.0)

PILOT_BLOCK = simPilot.RealisticSimPilot(
    name="Block STF",
    kp=0.10,
    kd=0.38,
    targetDolphin_v=targetCruise_v,
    targetCruise_v=targetCruise_v,
    n_max=1.2,
    n_min=0.8,
    w_pullThresh=2.0,
    w_pushThresh=0.0,
)

PILOT_SMOOTH = simPilot.RealisticSimPilot(
    name="SmoothOperator",
    kp=0.11965410743410854,
    kd=0.9135092815515045,
    targetDolphin_v=28.29905227678825,
    targetCruise_v=49.0,
    n_max=1.2,
    n_min=0.8,
    w_pullThresh=2.7776698384242495,
    w_pushThresh=-0.9200110433052546
)

PILOT_AGGRESSIVE = simPilot.RealisticSimPilot(
    name="AggroCraig",
    kp=0.037857535877599595,
    kd=0.20402355352215196,
    targetDolphin_v=28.67164663452786,
    targetCruise_v=49.0,
    n_max=2.0,
    n_min=0.5,
    w_pullThresh=2.7785569166199466,
    w_pushThresh=0.46876167327718576
)

PILOT_OPTIMIZED = simPilot.RealisticSimPilot(
    name="Maverick",
    kp=0.017606343202116122,
    kd=0.13675295649045438,
    targetDolphin_v=28.69461352158396,
    targetCruise_v=49.0,
    n_max=2.5,
    n_min=0.0,
    w_pullThresh=2.7816693254808236,
    w_pushThresh=0.5904410714865662
)

PILOT_CHEATER = simPilot.CheaterSimPilot(
    name="Cheater",
    x=simCheater.initialCheaterParameters(thermalCenter, thermalWidth)[0],
    n=simCheater.initialCheaterParameters(thermalCenter, thermalWidth)[1],
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
    global state
    state.t = 0.0 # s
    state.x = 0.0 # m
    state.z = 0.0 # m
    state.v = targetCruise_v # m/s
    state.dv_dt = 0.0
    state.n = 1.0 # m/s
    state.gamma = steadyStateGamma(state.v, state.n)

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

def advanceState(pilot: simPilot.SimPilot):
    # RK4 update
    global state

    # We have a simulation singularity at state.v = 0 where cl becomes infinite.
    # We must stop early on such singularities
    cl = commandedLiftCoefficient(state.n, max(1e-3, state.v))
    if cl > 5.0:
        raise SimulationError("Glider is deeply stalled")

    # 1. Update pilot state and control input
    localW = w(state.x)
    localShear = dw_dx(state.x)
    pilot.advance_state(state, localW, localShear)
    nCmd = pilot.n_cmd(state, localW, localShear)
    
    # 2. RK4 Intermediate steps
    # k1
    k1_x, k1_z, k1_v, k1_g, k1_n = derivatives(state.x, state.z, state.v, state.gamma, state.n, nCmd)
    
    # k2
    k2_x, k2_z, k2_v, k2_g, k2_n = derivatives(
        state.x + 0.5 * dt * k1_x,
        state.z + 0.5 * dt * k1_z,
        state.v + 0.5 * dt * k1_v,
        state.gamma + 0.5 * dt * k1_g,
        state.n + 0.5 * dt * k1_n,
        nCmd
    )
    
    # k3
    k3_x, k3_z, k3_v, k3_g, k3_n = derivatives(
        state.x + 0.5 * dt * k2_x,
        state.z + 0.5 * dt * k2_z,
        state.v + 0.5 * dt * k2_v,
        state.gamma + 0.5 * dt * k2_g,
        state.n + 0.5 * dt * k2_n,
        nCmd
    )
    
    # k4
    k4_x, k4_z, k4_v, k4_g, k4_n = derivatives(
        state.x + dt * k3_x,
        state.z + dt * k3_z,
        state.v + dt * k3_v,
        state.gamma + dt * k3_g,
        state.n + dt * k3_n,
        nCmd
    )
    
    # 3. Weighted state updates
    state.x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    state.z += (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    state.v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    # TODO: think careful about whether we should call derivatives() again to get an updated dv/dt
    state.dv_dt = k1_v
    state.gamma += (dt / 6.0) * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g)
    state.n += (dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
    
    state.t += dt

def energyAsHeight():
    return state.z + state.v ** 2 / (2 * g)

def detrendedEnergyHeight():
    # Adjust the total energy height for maccready
    w_mc = impliedMacCready(targetCruise_v)
    totalEnergy_height = state.z + state.v ** 2 / (2 * g) - w_mc * state.t

    # Steady cruise slope (m of energy height lost per m of horizontal distance)
    s_eff = (sinkRateInStillAir(targetCruise_v) + w_mc) / targetCruise_v
    # Detrended energy height
    baseline_E_h = - (s_eff * state.x)

    delta_E_h = (totalEnergy_height - baseline_E_h)
    return delta_E_h

def printState():
    v_kt = state.v * 1.94
    x_ft = state.x * 3.28
    z_ft = state.z * 3.28
    pitch_deg = state.gamma / math.pi * 180.0
    print(f'{state.t:.1f}\t{v_kt:.1f}\t{x_ft:.0f}\t{z_ft:.1f}\t{pitch_deg:.1f}\t{state.n:.1f}')

    #delta_E_h = detrendedEnergyHeight()

    #delta_E_h_ft = delta_E_h * 3.28
    #print(f'{x_ft:.0f}\t{delta_E_h_ft:.0f}')

def simulate(tMax, pilot: simPilot.SimPilot):
    initializeState()

    history = {"t": [], 
               "x": [], 
               "z": [], 
               "v": [], 
               "dv_dt": [],
               "gamma": [], 
               "n": [], 
               "n_cmd": [], 
               "E_h": [], 
               "E_h_detrended": []
               }
    history['simulationOK'] = True

    while state.t < tMax:
        history['t'].append(state.t)
        history['x'].append(state.x)
        history['z'].append(state.z)
        history['v'].append(state.v)
        history['dv_dt'].append(state.dv_dt)
        history['gamma'].append(state.gamma)
        history['n'].append(state.n)
        history['n_cmd'].append(pilot.n_cmd(state, w(state.x), dw_dx(state.x)))
        history['E_h'].append(energyAsHeight())
        history['E_h_detrended'].append(detrendedEnergyHeight())
        try:
            advanceState(pilot)
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

    if True:
        data = simulate(30.0, PILOT_OPTIMIZED)
        x_w = range(0,1400,10)
        y_w = [w(x) for x in x_w]
        y_dw = [10 * dw_dx(x) for x in x_w]
        figw = plt.figure()
        ax = figw.add_subplot()
        ax2 = ax.twinx()
        ax.plot(x_w, y_w)
        #ax2.plot(x_w, y_dw, color='red')
        ax2.plot(data['x'], data['n'], color='red')

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