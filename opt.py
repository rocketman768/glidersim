#!/usr/bin/env python3

import sim
import simCheater

from skopt import forest_minimize
from skopt.space import Real

import cma

def f(x, isCheater: bool = False):
    pilot = None

    if isCheater:
        xCheater_init, nCmdCheater_init = simCheater.initialCheaterParameters(sim.thermalWidth, sim.thermalCenter)
        pilot = sim.PilotProfile(
            name="TestCheater",
            kp=0,
            kd=0,
            targetDolphin_v=25.0,
            n_max=3.0,
            n_min=1.0,
            v_pullThresh=0,
            v_pushThresh=0,
            nCmd_cheater=nCmdCheater_init,
            nCmd_x_cheater=xCheater_init
        )
        pilot.nCmd_cheater = x
    else:
        pilot = sim.PilotProfile(
            name="TestPilot",
            kp=x[0],
            kd=x[1],
            targetDolphin_v=x[2],
            n_max=x[3],
            n_min=x[4],
            v_pullThresh=x[5],
            v_pushThresh=x[6]
        )

    try:
        data = sim.simulate(30.0, pilot)
    except:
        # Give some large penalty for breaking the simulation
        return -1e6

    # This is going to be something like 6 (m)
    term0 = data['E_h_detrended'][-1] - data['E_h_detrended'][0]
    # All pilots need to finish at the starting velocity
    term1 = abs(data['v'][-1] - data['v'][0])
    # All pilots need to finish at constant velocity
    term2 = abs(data['v'][-1] - data['v'][-2]) / (data['t'][-1] - data['t'][-2])

    return term0 - 0.1 * term1 - 0.5 * term2

def df_dx(x, isCheater: bool = False):
    dx = 0.001

    def df(n):
        x0 = x[0:n] + [x[n] - dx] + x[(n+1):]
        x1 = x[0:n] + [x[n] + dx] + x[(n+1):]
        return (f(x1, isCheater) - f(x0, isCheater)) / (2 * dx)

    return [df(n) for n in range(len(x))]

def optimizeCheater():
    numCheaterPoints = len(simCheater.initialCheaterParameters(sim.thermalCenter, sim.thermalWidth)[1])
    x = [1.0] * numCheaterPoints

    numCheaterPoints = len(x)

    def roughness(arg):
        sum = 0.0
        for ndx in range(0,len(arg)-1):
            sum += (arg[ndx+1] - arg[ndx])**2
        return sum / len(arg)
    def objFun(arg):
        x = [float(a) for a in arg]
        return -f(x, True) + 0.5 * roughness(x)

    print(objFun(x))
    options = {
        'bounds': [0.0, 3.0],  # Enforces physical load factor limits
        'popsize': 32,          # Slightly larger population for 55D
        'maxiter': 500
    }
    sigma0 = 0.05
    x_best, es = cma.fmin2(objFun, x, sigma0, options=options)
    print([float(a) for a in x_best])

def optimizePilot():
    #       [ 0,   1,    2,    3,    4,     5,     6]
    #       [ kp, kd, vdol, nmax, nmin, vpull, vpush]
    x0    = [0.10, 0.38, 28.0, 3.0, 0.0, 0, 0]
    x_min = [0.01, 0.0, 25.0, 1.01, 0.0, 0.0, 0.0]
    x_max = [0.2, 0.8, 49.0, 3.0, 0.9, 2.5, 2.5]

    def objFun(arg):
        x = [float(a) for a in arg]
        return -f(x)

    options = {
        'bounds': [x_min, x_max],
        'fixed_variables': {3: x0[3], 4: x0[4]},
        'popsize': 32,          # Slightly larger population for 55D
        'maxiter': 500
    }
    sigma0 = 0.05
    x_best, es = cma.fmin2(objFun, x0, sigma0, options=options)
    
    print(f'{objFun(x_best):.4f}')
    print([float(a) for a in x_best])

if __name__ == '__main__':
    optimizePilot()