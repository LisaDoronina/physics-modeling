from idlelib.debugobj_r import remote_object_tree_item

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from config import Params, ANGLES, V0, K2

def state_derivative(t, cur_state, modus: str):
    """cur_state = [x, y, vx, vy]
    returns производную вектора"""
    x, y, vx, vy = cur_state
    v = np.hypot(vx, vy) # модуль вектора скорости как длина гипотенузы

    if modus == 'none':
        ax = 0.0
        ay = -Params.g
    elif modus == 'lin':
        # F = -k * v
        # a = - k * v / m
        ax = - (Params.k1 / Params.m) * vx
        ay = -Params.g - (Params.k1 / Params.m) * vy
    elif modus == 'quadr':
        # F = -k2 * |v| * v
        # a = -(k2 / m) * |v| * v
        ax = -(Params.k2 / Params.m) * v * vx
        ay = -Params.g - (Params.k2 / Params.m) * v * vy
    else:
        print("unknown modus: ", modus)
        return None

    return [vx, vy, ax, ay]

def hit_ground(t, u):
    """returns y"""
    return u[1]

hit_ground.terminal  = True
hit_ground.direction = -1

def computing(modus):
    alpha = np.deg2rad(Params.alpha)
    vx0 = Params.v0 * np.cos(alpha)
    vy0 = Params.v0 * np.sin(alpha)

    cur_state = [Params.x0, Params.y0, vx0, vy0]

    sol = solve_ivp(
        fun=lambda t, u: state_derivative(t, u, modus),
        t_span=(0.0, Params.t_max),
        y0=cur_state,
        method="RK45", # метод для расчета, оценивает ошибку
        rtol=Params.rtol, # точность
        atol=Params.atol,
        events=hit_ground, # стопает если y = 0
        dense_output=True
    ) # sol.t = { массив моментов времени }
    # sol.y = { массив состояний в эти моменты }
    # sol.t_events[0] - массив времен приземления

    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0]
        t = np.linspace(0.0, t_end, 400) # сетка 400 точек от 0 до t_end
        Y = sol.sol(t) # информация о каждом состоянии
    else: # if не успел долететь за t_max
        t = sol.t
        Y = sol.y

    return {
        "t": t,
        "x": Y[0],
        "y": Y[1],
        "t_flight":  t[-1], # момент падения
        "x_range": Y[0][-1], # дальность полета
    }

def theory_none(t):
    """теория: свободный полёт, по параболе"""

    alpha = np.deg2rad(Params.alpha)
    vx0 = Params.v0 * np.cos(alpha)
    vy0 = Params.v0 * np.sin(alpha)
    x = Params.x0 + vx0 * t
    y = Params.y0 + vy0 * t - 0.5 * Params.g * t ** 2
    return x, y

def theory_lin(t):
    """теория: с сопротивлением"""

    gamma = Params.k1 / Params.m
    alpha = np.deg2rad(Params.alpha)
    vx0 = Params.v0 * np.cos(alpha)
    vy0 = Params.v0 * np.sin(alpha)

    if gamma == 0:
        return theory_none(t)

    x = Params.x0 + (vx0 / gamma) * (1 - np.exp(-gamma * t))
    y = (Params.y0 + (1.0 / gamma) * (vy0 + Params.g / gamma) * (1 - np.exp(-gamma * t)) - (Params.g / gamma) * t)

    return x, y

def computing_variant(mode, **overrides):
    saved = {k: getattr(Params, k) for k in overrides}

    for k, v in overrides.items():
        setattr(Params, k, v)

    try:
        return computing(mode)
    finally:
        for k, v in saved.items():
            setattr(Params, k, v)

def plot_none():
    """Без сопротивления: численно vs теория"""
    fig, ax = plt.subplots(figsize=(9, 5))

    r = computing("none")
    ax.plot(r["x"], r["y"], color="tab:green", lw=2, label="численно")

    t = np.linspace(0, r["t_flight"], 200)
    xa, ya = theory_none(t)
    ax.plot(xa, ya, "k--", lw=1.2, label="теория (парабола)")

    ax.set_xlabel("ДЛИНА, М")
    ax.set_ylabel("ВЫСОТА, М")
    ax.set_title("ТРАЕКТОРИЯ без сопротивления")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_linear():
    """Линейное сопротивление: численно vs теория"""
    fig, ax = plt.subplots(figsize=(9, 5))

    r = computing("lin")
    ax.plot(r["x"], r["y"], color="tab:blue", lw=2, label="численно")

    t = np.linspace(0, r["t_flight"], 200)
    xa, ya = theory_lin(t)
    ax.plot(xa, ya, "k--", lw=1.2, label="теория (F ∝ v)")

    ax.set_xlabel("ДЛИНА, М")
    ax.set_ylabel("ВЫСОТА, М")
    ax.set_title("ТРАЕКТОРИЯ при линейном сопротивлении ")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_quadratic():
    """Квадратичное сопротивление: численно"""
    fig, ax = plt.subplots(figsize=(9, 5))

    r_q = computing("quadr")
    ax.plot(r_q["x"], r_q["y"], color="tab:red", lw=2,
            label="Квадратичное сопротивление")

    r_n = computing("none")
    ax.plot(r_n["x"], r_n["y"], "--", color="tab:green", lw=2, label="Без сопротивления")

    r_l = computing("lin")
    ax.plot(r_l["x"], r_l["y"], "--", color="tab:blue", lw=2, label="Линейное сопротивление")

    ax.set_xlabel("ДЛИНА, М")
    ax.set_ylabel("ВЫСОТА, М")
    ax.set_title("ТРАЕКТОРИЯ для квадратичного сопротивления В СРВАВНЕНИИ")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_angles(mode):
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in ANGLES:
        r = computing_variant(mode, alpha=a)
        ax.plot(r["x"], r["y"], "--", lw=2, label=f"α={a}°")
    ax.set_xlabel("ДЛИНА, М")
    ax.set_ylabel("ВЫСОТА, М")
    modus = ""
    match mode:
        case "none":
            modus = "движения без сопротивления"
        case "lin":
            modus = "движения с сопротивлением"
        case "quadr":
            modus = "движения с лобовым сопротивлением"
    ax.set_title("РАЗНЫЕ УГЛЫ для {}".format(modus))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_range_vs_angle():
    """расстояние vs угол"""
    angles = np.linspace(5, 85, 60)
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, color in (("none","green"),("lin","blue"),("quadr","red")):
        ranges = []

        modus = ""
        match mode:
            case "none":
                modus = "Без сопротивления"
            case "lin":
                modus = "Линейное сопротивление"
            case "quadr":
                modus = "Квадратичное сопротивление"
            case _:
                modus = "???"

        for a in angles:
            r = computing_variant(mode, alpha=a)
            ranges.append(r["x_range"])
        ax.plot(angles, ranges, color=color, lw=2, label=modus)

    ax.axvline(45, color="gray", ls=":", lw=1)
    ax.set_xlabel("угол alpha, гр.")
    ax.set_ylabel("РАССТОЯНИЕ, м")
    ax.set_title("РАССТОЯНИЕ от угла")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

def plot_range_vs_v0():
    """расстояние vs v_0"""
    fig, ax = plt.subplots(figsize=(9, 5))
    v_list = [10, 20, 30, 40, 50]
    for mode, color in (("none","green"),("lin","blue"),("quadr","red")):
        Ls = [computing_variant(mode, v0=v)["x_range"] for v in v_list]
        ax.plot(v_list, Ls, "o-", color=color, lw=2, label=mode)
    ax.set_xlabel("v0, м/с")
    ax.set_ylabel("РАССТОЯНИЕ, М")
    ax.set_title("РАССТОЯНИЕ от начальной скорости")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

def plot_k2_vs_trajectory():
    """коэффициент лобового сопротивления vs расстояние"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = computing_variant("quadr", k2=k)
        ax.plot(r["x"], r["y"], lw=2, label=f"k = {k}")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title("ТРАЕКТОРИЯ при разных коэфициентах сопротивления")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def take_input():
    print("Начальные значения: m = 1.0, v0 = 30.0, alpha = 45.0")
    print("Хотите ввести другие значения? (y/n)")
    answer = input()
    if answer.upper() in ["Y", "YES"]:
        print("Введите массу, начальную скорость и угол в float: ")
        m = float(input())
        v0 = float(input())
        alpha = float(input())
        apply_input(m, v0, alpha)
    else:
        return

def apply_input(m, v0, alpha):
    Params.m = m
    Params.v0 = v0
    Params.alpha = alpha


def main():
    take_input()

    plots = [
        plot_none(),
        plot_linear(),
        plot_quadratic(),
        plot_angles("lin"),
        plot_range_vs_angle(),
        plot_range_vs_v0(),
        plot_k2_vs_trajectory(),
    ]
    for fig in plots:
        plt.show(block=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
