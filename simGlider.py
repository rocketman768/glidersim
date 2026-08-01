from abc import ABC, abstractmethod
import math

class SimGlider(ABC):
    @abstractmethod
    def cd(self, cl: float) -> float:
        '''Return the total coefficient of drag for given coefficient of lift'''
        pass

    @abstractmethod
    def S(self) -> float:
        '''Return the total wing area in m^2'''
        pass

    @abstractmethod
    def m(self) -> float:
        '''Total mass in kg'''
        pass

class SimJS3(SimGlider):
    def cd(self, cl: float) -> float:
        # 0 m/s = 524, 2 m/s = 3522
        # 100 kph = 1941, 200 kph = 6304

        # x, y (pixels) for 18m, 600 kg
        # 1553, 1621 - stalled
        # 1579, 1395 - still stalled
        # 1624, 1351 - barely not stalled
        # 1941, 1347 - 100 kph
        # 2378, 1367 - 110 kph
        # 2814, 1406 - 120 kph
        # 3249, 1468 - 130 kph (L/D = 57.8)
        # 3686, 1566 - 140 kph
        # 4123, 1681 - 150 kph
        # 4559, 1832 - 160 kph
        # 4996, 1991 - 170 kph
        # 5431, 2173 - 180 kph
        # 5868, 2376 - 190 kph
        # 6304, 2614 - 200 kph
        # 6740, 2896 - 210 kph
        # 7177, 3188 - 220 kph
        # 7613, 3541 - 230 kph
        # 7731, 3638

        # [cl, cd] assuming rho_air = 1.22 kg/m^3, g = 9.82 m/s^2
        _polarData = [
            [1.48427031054851,	0.0425802340063828],
            [1.46504229826138,	0.0331531895075337],
            [1.43263484361446,	0.030439722539908],
            [1.23201739712278,	0.0241577794203766],
            [1.01789924373309,	0.0185830364069275],
            [0.855436921083842,	0.0149789884790392],
            [0.729235784508286,	0.0126184458306205],
            [0.628621469722968,	0.0111476531889527],
            [0.547479629493661,	0.0100604457042712],
            [0.481229220910691,	0.0093727632079206],
            [0.426200160046064,	0.00876158022903776],
            [0.380291021070209,	0.00830092860415384],
            [0.341254357286754,	0.00792483458464983],
            [0.308004349280696,	0.00766855391685715],
            [0.279387319876873,	0.0075189403182471],
            [0.254527834803722,	0.00734292446462327],
            [0.232890896566655,	0.00727840126471054],
            [0.227508954186901,	0.00725351084849844],
        ]

        # cl too high
        if cl >= _polarData[0][0]:
            # Continue straight line in a stall
            return _polarData[0][1] + (_polarData[1][1] - _polarData[0][1]) / (_polarData[1][0] - _polarData[0][0]) * (cl - _polarData[0][0])

        # cl too low
        if cl <= _polarData[-1][0]:
            # Pure guess here...polar doesn't show the low end drop off
            return _polarData[-1][1] + (_polarData[-1][0] - cl) * 100e-4/0.1

        # Find the index in the table
        leftNdx = 0
        for ndx, pair in enumerate(_polarData):
            leftNdx = ndx
            if pair[0] <= cl:
                break

        #print(f'{cl} {leftNdx}')
        # Linear interpolation between measured points
        clLeft = _polarData[leftNdx][0]
        clRight = _polarData[leftNdx-1][0]
        cdLeft = _polarData[leftNdx][1]
        cdRight = _polarData[leftNdx-1][1]
        a = (cl - clLeft) / (clRight - clLeft)

        return a * cdRight + (1.0 - a) * cdLeft

    def S(self) -> float:
        return 10.0

    def m(self) -> float:
        return 600.0

class SimNixus(SimGlider):
    def __init__(self):
        span = 18.0
        self._AR = span * span / self.S()
        self._eOswald = 0.95

    def cd(self, cl: float) -> float:
        return self._cd0(cl) + self._cdi(cl)

    def S(self) -> float:
        return 10.0

    def m(self) -> float:
        return 500.0

    # Valid for cl in [0.281, 1.33]
    def _cd0(self, cl):
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

    def _cdi(self, cl):
        return cl * cl / (math.pi * self._AR * self._eOswald)