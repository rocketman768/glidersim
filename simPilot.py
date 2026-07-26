from abc import ABC, abstractmethod
from dataclasses import dataclass
import simConstants

@dataclass
class SimState:
    t: float # s
    x: float # m
    z: float # m
    v: float # m/s
    dv_dt: float # m/s/s
    gamma: float # radians (note: initialize this to proper steady-state for x/z/v)
    n: float # Gs of acceleration. Control input.

class SimPilot(ABC):
    @abstractmethod
    def n_cmd(self, state: SimState, w: float, dw_dx: float) -> float:
        """
        Returns the currently-commanded vertical acceleration in gs based on
        the simulation state. No pilot state mutation here.
        """
        pass

    @abstractmethod
    def advance_state(self, state: SimState, w: float, dw_dx: float) -> None:
        """
        State mutation. Advances the internal pilot states.
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset any internal state to inifial state.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[float]:
        """Returns a list of optimizable parameters"""
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, x: list[float]) -> None:
        """Set the optimizable parameters"""
        pass

    @abstractmethod
    def parameter_limits(self) -> tuple[list[float], list[float]]:
        """Return xMin[n] and xMax[n] that give the bounds for parameters[n]"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The identifier name of this specific autopilot implementation."""
        pass

    def validate_dimensions(self) -> bool:
        """Helper to verify that parameters and limits match sizes."""
        x_min, x_max = self.parameterLimits()
        p_len = len(self.parameters)
        
        if not (p_len == len(x_min) == len(x_max)):
            raise ValueError(
                f"Dimension mismatch! Params: {p_len}, Min Bounds: {len(x_min)}, Max Bounds: {len(x_max)}"
            )
        return True

class CheaterSimPilot(SimPilot):
    def __init__(self, name: str, x: list[float] = None, n: list[float] = None):
        self._name = name
        DX = 60
        #x_max = thermalCenter + thermalWidth * 4
        #numPoints = int(x_max / DX)
    
        #x = [ndx * DX for ndx in range(numPoints)]
        #n = [1.0 for x0 in x]

        if x:
            self._x = x
            if n:
                self._n = n
            else:
                self._n = [1.0] * len(x)
        else:
            self._n = [1.841312852415778, 1.2216451930760384, 0.7892138367987785, 0.6767995254542043, 0.6979767060147137, 0.32926575649896056, 1.0329887172102887, 2.9999978895336814, 1.6908461874872247, 0.0950062827846719, 0.704473319446135, 0.6087591382843389, 0.775874157712199, 1.034032097363144, 1.064324582666054, 1.0814679097859985, 1.069300320862634, 1.0508063812660617, 1.0380047398677834, 1.0236277051547364, 1.0181877107555457, 1.0129262630133762, 1.0155849938031376, 1.0094191317937597, 1.0597688703652854, 1.0924087250224497, 1.0974392443946523]
            self._x = [ndx * DX for ndx in range(len(self._n))]

    def n_cmd(self, state: SimState, w: float, dw_dx: float) -> float:
        x = self._x
        n = self._n
    
        def findNdx():
            for ndx, x0 in enumerate(x):
                if x0 > state.x:
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
        a = (state.x - xleft) / (xright - xleft)
        nleft = n[ndx]
        nright = n[ndx+1]
    
        return (1-a) * nleft + a * nright

    def advance_state(self, state: SimState, w: float, dw_dx: float) -> None:
        pass

    def reset_state(self):
        pass

    @property
    def parameters(self) -> list[float]:
        return self._n

    @parameters.setter
    def parameters(self, x: list[float]) -> None:
        self._n = x

    @property
    def name(self) -> str:
        return self._name

    def parameter_limits(self) -> tuple[list[float], list[float]]:
        minG = -1.0
        maxG = 5.0

        numel = len(self._n)
        return ([minG] * numel, [maxG] * numel)

class RealisticSimPilot(SimPilot):
    def __init__(self,
        name: str,
        kp: float,      # Speed error gain (g's per m/s speed error)
        kd: float,      # Acceleration damping gain (g's per m/s^2 acceleration)
        targetDolphin_v: float, # Target dolphin speed (m/s)
        targetCruise_v: float,
        n_max: float,   # Upper load factor limit (g)
        n_min: float,   # Lower load factor limit (g)
        w_pullThresh: float, # Pilot pulls once the thermal is predicted stronger than this (m/s)
        w_pushThresh: float, # Pilot pushes once the thermal is predicted weaker than this (m/s)
        allowLoadFactorLimitTuning: bool = False # If True, optimizer can tune n_max/n_min
    ):
        self._name = name
        self._kp = kp
        self._kd = kd
        self._targetDolphin_v = targetDolphin_v
        self._targetCruise_v = targetCruise_v
        self._n_max = n_max
        self._n_min = n_min
        self._w_pullThresh = w_pullThresh
        self._w_pushThresh = w_pushThresh
        self.allowLoadFactorLimitTuning = allowLoadFactorLimitTuning

        self.reset_state()

    def __str__(self):
        return f'''
simPilot.RealisticSimPilot(
    name="{self._name}",
    kp={self._kp},
    kd={self._kd},
    targetDolphin_v={self._targetDolphin_v},
    targetCruise_v={self._targetCruise_v},
    n_max={self._n_max},
    n_min={self._n_min},
    w_pullThresh={self._w_pullThresh},
    w_pushThresh={self._w_pushThresh}
)
        '''

    def n_cmd(self, state: SimState, w: float, dw_dx: float) -> float:
        # 0.2 per 2 m/s
        kp = self._kp
        # Tune to prevent overshoot
        kd = self._kd
        n_max = self._n_max
        n_min = self._n_min
        w_pullThresh = self._w_pullThresh
        w_pushThresh = self._w_pushThresh
        # TODO: move to pilot profile?
        w_maxPull = 2.5 # m/s

        if self._state in ('CRUISE', 'PUSHOVER'):
            target_v = self._targetCruise_v
        else:
            target_v = self._targetDolphin_v
        #if pilot.name == 'Maverick':
        #    print(f'{state.pilot} {localShear:.4f} {state.v:.1f}')
    
        dw_dt = dw_dx * state.v
        if self._state in ('PULLUP', 'PUSHOVER') and w > 0:
            # Want:
            #   n(t) = k1 + k2 * w(t)
            # So:
            #   dn/dt = k2 dw/dt.
            # Also, from state equations,
            #   dn/dt = (n_cmd - n) / tau.
            # So:
            #   k2 dw/dt = (n_cmd - n) / tau
            #   k2 dw/dt = (n_cmd - (k1 + k2 w(t))) / tau
            #   tau k2 dw/dt = n_cmd - k1 - k2 w(t)
            #   n_cmd = k1 + k2 w(t) + tau * k2 * dw/dt
            # So:
            #   k3 = tau * k2
            # ...for perfect lag cancellation.
    
            # k1 is how much we push outside of the thermal (when w an dw/dt are 0)
            k1 = n_min
            # We want maximum pulling at the core (w = maxW)
            k2 = (n_max - n_min)/w_maxPull
            # k3 is computed to cancel lag as shown above
            k3 = k2 * simConstants.tau_n
    
            n_cmd = k1 + k2 * w + k3 * dw_dt
        else:
            # Speed control with MacCready target modulation
            # target_v decreases in lift (w > 0) and increases in sink (w < 0)
    
            # The sensitivity is how much you should speed up in m/s if there is 1 m/s of sink
            # at the current airspeed based on maccready analysis of the polar
            maccreadySensitivity = 6.0
    
            target_v_dynamic = target_v - maccreadySensitivity * w
    
            # I know we are double-adding terms proportional to localW here, but
            # kp * maccreadySensitivity will simply be to low to make quick airspeed
            # changes, and it's simply easier to think about first that we should adjust
            # the target speed, and THEN do some pushing based on the vario.
    
            # I want continuity between the two controllers at the pull threshold.
            # Assuming the velocity terms are close to 0...
            # 1.0 + kw * w_pullThresh = k1 + k2 * w_pullThresh + k3 * dw_dt(w_pullThresh)
            # Assuming dw_dt \approx 0
            # kw = k2 + (k1 - 1.0) / w_pullThresh
            # There is no actual w_pullThresh since we pull based on shear now, so just
            # assume it's like 2 m/s or ignore it completely and then kw = k2
    
            k2 = (n_max - n_min)/w_maxPull
            kw = k2 + (n_min - 1.0) / w_pullThresh
    
            n_cmd = 1.0 + kp * (state.v - target_v_dynamic) + kd * state.dv_dt + kw * w
        
        # Clamp
        return max(n_min, min(n_max, n_cmd))

    def advance_state(self, state: SimState, w: float, dw_dx: float) -> None:
        # Predict thermal strength tau_n seconds in the future.
        w_predicted = w + state.v * dw_dx * simConstants.tau_n

        # Pull-up trigger on entry gradient
        if w_predicted > self._w_pullThresh and self._state == 'CRUISE':
            self._state = 'PULLUP'
        # Stop pulling if airspeed drops to target dolphin speed
        elif self._state == 'PULLUP' and state.v <= self._targetDolphin_v:
            self._state = 'COAST'  # Holds 1.0g through the core
        # Push-over trigger on exit shear gradient
        elif w_predicted < self._w_pushThresh and self._state in ('PULLUP', 'COAST'):
            self._state = 'PUSHOVER'
        # Return to cruise once airspeed is fully recovered
        elif self._state == 'PUSHOVER' and state.v >= self._targetCruise_v:
            self._state = 'CRUISE'

    def reset_state(self):
        self._state = 'CRUISE'

    @property
    def parameters(self) -> list[float]:
        return [
            self._kp,
            self._kd,
            self._targetDolphin_v,
        ] + ([
            self._n_max,
            self._n_min
        ] if self.allowLoadFactorLimitTuning else []) + [
            self._w_pullThresh,
            self._w_pushThresh,
        ]

    @parameters.setter
    def parameters(self, x: list[float]) -> None:
        [
            self._kp,
            self._kd,
            self._targetDolphin_v,
        ] = x[0:3]
        if self.allowLoadFactorLimitTuning:
            [
                self._n_max,
                self._n_min,
                self._w_pullThresh,
                self._w_pushThresh,
            ] = x[3:]
        else:
            [
                self._w_pullThresh,
                self._w_pushThresh,
            ] = x[3:]

    @property
    def name(self) -> str:
        return self._name

    def parameter_limits(self) -> tuple[list[float], list[float]]:
        xmin = [
            0.01,
            0.0,
            23.0,
        ] + ([
            1.1,
            -1.0
        ] if self.allowLoadFactorLimitTuning else []) + [
            0,
            -1.0,
        ]

        xmax = [
            1.0,
            4.0,
            70.0,
        ] + ([
            5.0,
            0.95
        ] if self.allowLoadFactorLimitTuning else []) + [
            10.0,
            10.0,
        ]
        return (xmin, xmax)