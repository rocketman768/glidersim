#!/usr/bin/env python3

import sim

def f(x):
    pilot = sim.PilotProfile(
        name="TestPilot",
        kp=x[0],
        kd=x[1],
        targetDolphin_v=x[2],
        n_max=x[3],
        n_min=x[4],
        x_lookahead=50,
    )
    data = sim.simulate(30.0, pilot)
    return data['E_h_detrended'][-1] - data['E_h_detrended'][0]

if __name__ == '__main__':
    x = [0.10, 0.38, 28.0, 2.0, 0.5]
    x_min = [0.01, 0.0, 25.0, 1.01, 0.0]
    x_max = [0.2, 0.8, 49.0, 3.0, 0.99]

    n_samples = 10
    n_rounds = 10

    for round in range(n_rounds):
        # One round of coordinate descent
        for coordinate in range(len(x)):
            print(f'== x[{coordinate}] ==')
            x_best = x[coordinate]
            y_best = -float("inf")
            for n in range(n_samples):
                x[coordinate] = x_min[coordinate] + (n / float(n_samples - 1)) * (x_max[coordinate] - x_min[coordinate])
                y = f(x)
                if y > y_best:
                    x_best = x[coordinate]
                    y_best = y
                print(f'{x[coordinate]:.2f}: {y:.4f}')
            x[coordinate] = x_best

    print(x)