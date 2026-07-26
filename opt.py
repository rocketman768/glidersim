#!/usr/bin/env python3

import sim
import simCheater
import simPilot

from skopt import forest_minimize
from skopt.space import Real

import cma

def f(x, pilot: simPilot.SimPilot):
    try:
        pilot.reset_state()
        pilot.parameters = x
        data = sim.simulate(30.0, pilot)
    except:
        # Give some large penalty for breaking the simulation
        return -1e6

    # This is going to be something like 6 (m)
    term0 = data['E_h_detrended'][-1] - data['E_h_detrended'][0]
    # All pilots need to finish at the starting velocity
    term1 = abs(data['v'][-1] - data['v'][0])
    # All pilots need to finish at constant velocity
    term2 = abs(data['dv_dt'][-1])

    return term0 - 0.1 * term1 - 0.5 * term2

def optimizedCheater():
    x = simCheater.initialCheaterParameters(sim.thermalCenter, sim.thermalWidth)[0]
    numCheaterPoints = len(simCheater.initialCheaterParameters(sim.thermalCenter, sim.thermalWidth)[1])
    n0 = [1.0] * numCheaterPoints

    pilot = simPilot.CheaterSimPilot('Cheater', x, n0)

    numCheaterPoints = len(n0)

    def roughness(arg):
        sum = 0.0
        for ndx in range(0,len(arg)-1):
            sum += (arg[ndx+1] - arg[ndx])**2
        return sum / len(arg)
    def objFun(arg):
        n = [float(a) for a in arg]
        return -f(n, pilot) + 0.5 * roughness(n)

    print(objFun(n0))
    options = {
        'bounds': [0.0, 3.0],  # Enforces physical load factor limits
        'popsize': 32,          # Slightly larger population for 55D
        'maxiter': 500
    }
    sigma0 = 0.05
    n_best, es = cma.fmin2(objFun, n0, sigma0, options=options)
    print([float(a) for a in n_best])

def optimizedPilot(pilot: simPilot.SimPilot):
    def objFun(arg):
        x = [float(a) for a in arg]
        return -f(x, pilot)

    print(pilot.parameter_limits())
    print(pilot.parameters)
    (x_min, x_max) = pilot.parameter_limits()
    options = {
        'bounds': [x_min, x_max],
        'popsize': 32,          # Slightly larger population for 55D
        'maxiter': 500
    }
    sigma0 = 0.05
    x_best, es = cma.fmin2(objFun, pilot.parameters, sigma0, options=options)

    pilot.parameters = [float(a) for a in x_best]
    
    print(f'{objFun(x_best):.4f}')
    print(pilot.parameters)
    return pilot

if __name__ == '__main__':
    opt_pilots = []
    #for pilot in (sim.PILOT_BLOCK, sim.PILOT_SMOOTH, sim.PILOT_AGGRESSIVE, sim.PILOT_OPTIMIZED):
    for pilot in [sim.PILOT_OPTIMIZED]:
        pilot.allowLoadFactorLimitTuning = True
        pilot = optimizedPilot(pilot)
        opt_pilots.append(pilot)

    for pilot in opt_pilots:
        print(pilot)