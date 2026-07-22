#!/usr/bin/env python3

import sim

def f(x):
    pilot = sim.PilotProfile(
        name="Aggro (2.0g)",
        kp=0.10,
        kd=0.38,
        n_max=2.0,
        n_min=x,
        x_lookahead=50,
    )
    data = sim.simulate(30.0, pilot)
    return data['E_h_detrended'][-1] - data['E_h_detrended'][0]

if __name__ == '__main__':
    for n in range(10):
        x = n/10.0 * 0.7
        y = f(x)
        print(f'{x:.2f}: {y:.4f}')