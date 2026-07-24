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
    n_cmdMin = 0.0
    n_cmdMax = 3.0

    n_samples = 60
    n_rounds = 10

    numCheaterPoints = len(simCheater.initialCheaterParameters(sim.thermalCenter, sim.thermalWidth)[1])
    #x = simCheater.initialCheaterParameters(sim.thermalCenter, sim.thermalWidth)[1]
    #x = [0.0, 0.0, 0.9661016949152541, 1.0169491525423728, 0.9661016949152541, 1.0169491525423728, 0.9661016949152541, 0.9661016949152541, 1.0169491525423728, 0.9661016949152541, 1.0169491525423728, 0.9661016949152541, 1.1186440677966103, 1.271186440677966, 1.4745762711864407, 1.3728813559322033, 1.2203389830508475, 1.1694915254237288, 1.0169491525423728, 0.6101694915254238, 0.711864406779661, 0.6610169491525424, 0.7627118644067796, 0.711864406779661, 1.0169491525423728, 1.0169491525423728, 1.0169491525423728, 1.0169491525423728, 0.9661016949152541, 0.9661016949152541, 1.0677966101694916, 1.0677966101694916, 1.0169491525423728, 1.0169491525423728, 1.0169491525423728, 0.9661016949152541, 1.0169491525423728, 0.9661016949152541, 0.9661016949152541, 0.9661016949152541, 0.9661016949152541, 0.8135593220338984, 0.7627118644067796, 0.8135593220338984, 0.8135593220338984, 0.864406779661017, 0.864406779661017, 0.9152542372881356, 0.9152542372881356, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x = [1.0] * numCheaterPoints

    numCheaterPoints = len(x)

    if True:
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
        return

    # Let the PID parameters be optimized at the end
    coordsToOptimize = range(numCheaterPoints)

    for round in range(n_rounds):
        print(f'= Round {round} ==')
        # One round of coordinate descent
        for coordinate in coordsToOptimize:
            print(f'\t== x[{coordinate}] ==')
            x_best = x[coordinate]
            y_best = -float("inf")
            for n in range(n_samples):
                x_min = n_cmdMin
                x_max = n_cmdMax
                x[coordinate] = x_min + (n / float(n_samples - 1)) * (x_max - x_min)
                y = f(x, True)
                if y > y_best:
                    x_best = x[coordinate]
                    y_best = y
            x[coordinate] = x_best
            print(f'\t\t{x_best:.2f}: {y_best:.4f}')
        print(f'\t{f(x, True):.4f}: {x}')

    print(f'{f(x, True):.4f}: {x}')

def optimizePilot():
    #       [ 0,   1,    2,    3,    4,     5,     6]
    #       [ kp, kd, vdol, nmax, nmin, vpull, vpush]
    x     = [0.10, 0.38, 28.0, 1.2, 0.8, 0, 0]
    x_min = [0.01, 0.0, 25.0, 1.01, 0.0, 0.0, 0.0, 150, 300]
    x_max = [0.2, 0.8, 49.0, 3.0, 0.90, 2.5, 2.5, 450, 800]

    n_samples = 20
    n_rounds = 10

    # Let the PID parameters be optimized at the end

    # Maverick: optimize everything, no cheating
    #coordsToOptimize = (2,3,4,5,6,0,1)

    # SmoothOperator
    x = [0.10, 0.38, 28.0, 3.0, 0.0, 0.5, 0.5]
    coordsToOptimize = (2,5,6,0,1)

    # Cheater
    #x.append(350)
    #x.append(550)
    #coordsToOptimize = (2,3,4,7,8,0,1)

    for round in range(n_rounds):
        print(f'= Round {round} ==')
        # One round of coordinate descent
        for coordinate in coordsToOptimize:
            print(f'\t== x[{coordinate}] ==')
            x_best = x[coordinate]
            y_best = -float("inf")
            for n in range(n_samples):
                x[coordinate] = x_min[coordinate] + (n / float(n_samples - 1)) * (x_max[coordinate] - x_min[coordinate])
                y = f(x, False)
                if y > y_best:
                    x_best = x[coordinate]
                    y_best = y
            x[coordinate] = x_best
            print(f'\t\t{x_best:.2f}: {y_best:.4f}')
        print(f'\t{f(x):.4f}: {x}')
        

    print(f'{f(x):.4f}: {x}')

if __name__ == '__main__':
    optimizeCheater()