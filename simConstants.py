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

# Simulation constants
dt = 0.1 # s
targetCruise_v = 49.0 # m/s
thermalWidth = 300.0 # m
thermalVelocity = 2.5 # m/s
thermalCenter = 450.0 # m