#!/usr/bin/env python3

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

# State vars
state_t = None # s
state_x = None # m
state_z = None # m
state_v = None # m/s
state_gamma = None # radians (note: initialize this to proper steady-state for x/z/v)

# Control inputs
n = 1.0 # Gs of acceleration

# Simulation constants
dt = 0.1 # s

# Pitch angle for given speed and loading
def steadyStateGamma(v, n_cmd):
    cl = commandedLiftCoefficient(n_cmd, v)
    cd = cd0(cl) + cdi(cl)
    drag = q(v) * S * cd
    return math.asin(-drag / (m * g))

def initializeState():
    global state_t, state_x, state_z, state_v, state_gamma
    state_t = 0.0 # s
    state_x = 0.0 # m
    state_z = 0.0 # m
    state_v = 49.0 # m/s
    state_gamma = steadyStateGamma(state_v, 1.0)

# Air motion model in vertical m/s
def w(x):
    thermalWidth = 300 # m
    thermalVelocity = 2.5 # m/s
    if x < 0 or x > thermalWidth:
        return 0.0
    else:
        return thermalVelocity

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

def derivatives(x, z, v, gamma, n0):
    """Calculates [dx/dt, dz/dt, dv/dt, dgamma/dt] for a given state."""
    cl = commandedLiftCoefficient(n0, v)
    cd = cd0(cl) + cdi(cl)
    drag = q(v) * S * cd
    
    dx_dt = v * math.cos(gamma)
    dz_dt = v * math.sin(gamma) + w(x)
    dv_dt = - (drag / m) - g * math.sin(gamma)
    dgamma_dt = (g / v) * (n0 - math.cos(gamma))
    
    return dx_dt, dz_dt, dv_dt, dgamma_dt

def controlUpdate():
    # 0.2 per 2 m/s
    kp = 0.2 / 2.0
    # Tune to prevent overshoot
    kd = 0.38
    n_max = 1.2
    n_min = 0.8

    # We need the velocity derivative
    _, _, v_dot, _ = derivatives(state_x, state_z, state_v, state_gamma, n)

    target_v = 28.0 if w(state_x) > 0 else 49.0    
    n_cmd = 1.0 + kp * (state_v - target_v)
    # Damping directly on the velocity derivative requires a lot of aero calcs
    n_cmd += kd * v_dot
    
    return max(n_min, min(n_max, n_cmd))

def advanceState():
    global n, state_t, state_x, state_z, state_v, state_gamma
    # RK4 update

    # 1. Update control input based on current state
    n = controlUpdate()
    
    # 2. RK4 Intermediate steps
    # k1
    k1_x, k1_z, k1_v, k1_g = derivatives(state_x, state_z, state_v, state_gamma, n)
    
    # k2
    k2_x, k2_z, k2_v, k2_g = derivatives(
        state_x + 0.5 * dt * k1_x,
        state_z + 0.5 * dt * k1_z,
        state_v + 0.5 * dt * k1_v,
        state_gamma + 0.5 * dt * k1_g,
        n
    )
    
    # k3
    k3_x, k3_z, k3_v, k3_g = derivatives(
        state_x + 0.5 * dt * k2_x,
        state_z + 0.5 * dt * k2_z,
        state_v + 0.5 * dt * k2_v,
        state_gamma + 0.5 * dt * k2_g,
        n
    )
    
    # k4
    k4_x, k4_z, k4_v, k4_g = derivatives(
        state_x + dt * k3_x,
        state_z + dt * k3_z,
        state_v + dt * k3_v,
        state_gamma + dt * k3_g,
        n
    )
    
    # 3. Weighted state updates
    state_x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    state_z += (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    state_v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    state_gamma += (dt / 6.0) * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g)
    
    state_t += dt

def printState():
    v_kt = state_v * 1.94
    x_ft = state_x * 3.28
    z_ft = state_z * 3.28
    pitch_deg = state_gamma / math.pi * 180.0
    print(f'{state_t:.1f}\t{v_kt:.1f}\t{x_ft:.0f}\t{z_ft:.1f}\t{pitch_deg:.1f}')

if __name__ == '__main__':
    initializeState()
    while state_t < 30.0:
        printState()
        advanceState()