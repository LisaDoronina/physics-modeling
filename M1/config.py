from dataclasses import dataclass
from typing import List


@dataclass
class Params:
    m: float = 1.0 # масса камня, кг
    g: float = 9.81 # ускорение свободного падения
    k1: float = 0.5 # коэф сопротивления, вязкое трение
    k2: float = 0.05 # коэф сопротивления, лобовое трение

    x0: float = 0.0
    y0: float = 0.0
    v0: float = 30.0
    alpha: float = 45.0

    t_max: float = 20.0
    rtol: float = 1e-8
    atol: float = 1e-10

    modus: str = "none" # режим сопротивления: "none", "lin", "quadr"


ANGLES: List[float] = [15, 30, 45, 60, 75]

V0: List[float] = [10, 20, 30, 40, 50]

K2: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2]