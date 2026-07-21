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
state_n = 1.0 # Gs of acceleration. Control input.

# Simulation constants
dt = 0.1 # s

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
    state_v = 49.0 # m/s
    state_n = 1.0 # m/s
    state_gamma = steadyStateGamma(state_v, state_n)

def w_box(x):
    thermalWidth = 300 # m
    thermalVelocity = 2.5 # m/s
    if x < 0 or x > thermalWidth:
        return 0.0
    else:
        return thermalVelocity

def w_allen(x):
    thermalWidth = 300.0 # m
    thermalVelocity = 2.5 # m/s

    x_c = thermalWidth / 2       # Center of thermal
    r0 = thermalWidth / 3        # Radius of zero-lift crossover
    w_peak = thermalVelocity     # Peak core lift (m/s)
    
    r = abs(x - x_c)
    norm_r = r / r0
    
    # Allen formula
    return w_peak * (1.0 - norm_r**2) * math.exp(-(norm_r**2))

# Air motion model in vertical m/s
def w(x):
    return w_allen(x)

def dw_dx(x):
    """Central difference derivative of w(x)."""
    dx_step = 0.01
    return (w(x + dx_step) - w(x - dx_step)) / (2.0 * dx_step)

def q(v):
    return 0.5 * rho * v * v

def commandedLiftCoefficient(n0, v):
    cl = n0 * m * g / (q(v) * S)
    # enforce model constraints...no flow separation
    if cl > 1.33:
        cl = 1.33
    if cl < 0.281:
        cl = 0.281
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

    return v

def cdi(cl):
    return cl * cl / (math.pi * AR * eOswald)

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

def controlUpdate():
    # 0.2 per 2 m/s
    kp = 0.2 / 2.0
    # Tune to prevent overshoot
    kd = 0.38
    n_max = 1.2
    n_min = 0.8

    # We need the velocity derivative. (pass dummy 0.0 for n_cmd since dv/dt doesn't use it)
    _, _, v_dot, _, _ = derivatives(state_x, state_z, state_v, state_gamma, state_n, 0.0)

    target_v = 28.0 if w(state_x) > 0 else 49.0 
    # Proportional term   
    n_cmd = 1.0 + kp * (state_v - target_v)
    # derivative term
    n_cmd += kd * v_dot
    # Clamp
    return max(n_min, min(n_max, n_cmd))

def advanceState():
    global state_t, state_x, state_z, state_v, state_gamma, state_n
    # RK4 update

    # 1. Update control input based on current state
    n_cmd = controlUpdate()
    
    # 2. RK4 Intermediate steps
    # k1
    k1_x, k1_z, k1_v, k1_g, k1_n = derivatives(state_x, state_z, state_v, state_gamma, state_n, n_cmd)
    
    # k2
    k2_x, k2_z, k2_v, k2_g, k2_n = derivatives(
        state_x + 0.5 * dt * k1_x,
        state_z + 0.5 * dt * k1_z,
        state_v + 0.5 * dt * k1_v,
        state_gamma + 0.5 * dt * k1_g,
        state_n + 0.5 * dt * k1_n,
        n_cmd
    )
    
    # k3
    k3_x, k3_z, k3_v, k3_g, k3_n = derivatives(
        state_x + 0.5 * dt * k2_x,
        state_z + 0.5 * dt * k2_z,
        state_v + 0.5 * dt * k2_v,
        state_gamma + 0.5 * dt * k2_g,
        state_n + 0.5 * dt * k2_n,
        n_cmd
    )
    
    # k4
    k4_x, k4_z, k4_v, k4_g, k4_n = derivatives(
        state_x + dt * k3_x,
        state_z + dt * k3_z,
        state_v + dt * k3_v,
        state_gamma + dt * k3_g,
        state_n + dt * k3_n,
        n_cmd
    )
    
    # 3. Weighted state updates
    state_x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    state_z += (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    state_v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    state_gamma += (dt / 6.0) * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g)
    state_n += (dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
    
    state_t += dt

def printState():
    v_kt = state_v * 1.94
    x_ft = state_x * 3.28
    z_ft = state_z * 3.28
    pitch_deg = state_gamma / math.pi * 180.0
    #print(f'{state_t:.1f}\t{v_kt:.1f}\t{x_ft:.0f}\t{z_ft:.1f}\t{pitch_deg:.1f}')

    totalEnergy_height = state_z + state_v ** 2 / (2 * g)
    totalEnergy_height_ft = totalEnergy_height * 3.28
    print(f'{x_ft:.0f}\t{totalEnergy_height_ft:.0f}')

if __name__ == '__main__':
    initializeState()
    while state_t < 30.0:
        printState()
        advanceState()