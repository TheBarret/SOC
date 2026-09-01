"""
Fourier Signal Engine, Dry Test Bench
=======================================

Isolates the TX / operator-chain / RX model from the spec,
one stage at a time, so each mathematical claim can be watched rather than assumed.

TX (source shape, §2 Static Generator)
-> Phase Shift        R_theta        (§4B)
-> Frequency Shift     H_m           (§4C)
-> Spectral Filter / Gain   W        (§4D)
-> DC-only boost (test probe, isolates C0 from "shape")
-> Attenuation + Noise over distance d   (§5.2)
-> Power Clamp to P_max                  (§5.1)
-> RX: eta and y_RX  (§3)

Controls
--------
  Target shape (T):     1=circle  2=triangle  3=square  4=star
  Source shape (TX):    Q=circle  W=triangle  E=square  R=star
  Phase theta:           LEFT / RIGHT
  Gain (uniform w_k):    UP / DOWN
  Freq shift m:          [ / ]
  Filter mode:           F   (cycles none -> low-pass -> high-pass)
  Filter cutoff K:       - / =
  DC-only boost |C0|:    Z / X   <-- triggers the y_RX exploit
  Distance d:            , / .
  Noise (thermal):       N   (toggle)
  Power clamp P_max:     C   (toggle)
  Reset all parameters:  BACKSPACE
  Quit:                  ESC
"""

import math
import sys

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 760
PANEL = 460                 # size of the shape-plot square
N = 8                        # harmonics k = -N .. N
M = 300                      # sample resolution for curve <-> coeff transform
GAMMA = 1.0
ETA_THRESH = 0.85
P_MAX = 40.0
ALPHA = 0.15                 # attenuation coefficient
NOISE_STD = 0.02
FPS = 60

WHITE = (235, 235, 235)
GREY = (110, 110, 110)
DIM = (70, 70, 70)
RED = (220, 90, 90)
CYAN = (90, 200, 220)
YELLOW = (230, 200, 90)
GREEN = (110, 210, 140)
BG = (18, 18, 22)

# ---------------------------------------------------------------------------
# §1/§2, Shapes and the coefficient <-> curve transform
# ---------------------------------------------------------------------------

def gen_circle():
    t = np.linspace(0, 1, M, endpoint=False)
    return np.exp(1j * 2 * np.pi * t)


def gen_polygon(sides):
    t = np.linspace(0, 1, M, endpoint=False)
    theta = t * 2 * np.pi
    seg = 2 * np.pi / sides
    phi = np.mod(theta, seg) - seg / 2
    r = np.cos(seg / 2) / np.cos(phi)
    return r * np.exp(1j * theta)


def gen_star(points_n=5, inner_ratio=0.45):
    t = np.linspace(0, 1, M, endpoint=False)
    theta = t * 2 * np.pi
    seg = np.pi / points_n
    phi = np.mod(theta, 2 * seg)
    tri = np.abs(phi / seg - 1.0)          # triangle wave 1 -> 0 -> 1
    r = inner_ratio + (1 - inner_ratio) * (1 - tri)
    return r * np.exp(1j * theta)


SHAPES = {
    "circle": gen_circle,
    "triangle": lambda: gen_polygon(3),
    "square": lambda: gen_polygon(4),
    "star": gen_star,
}
SOURCE_KEYMAP = {pygame.K_q: "circle", pygame.K_w: "triangle",
                 pygame.K_e: "square", pygame.K_r: "star"}
TARGET_KEYMAP = {pygame.K_1: "circle", pygame.K_2: "triangle",
                 pygame.K_3: "square", pygame.K_4: "star"}


def k_index(k):
    return k + N


def curve_to_coeffs(z_samples):
    """Direct application of the DFT: C_k are the Fourier coefficients of
    the sampled closed curve. See spec §1 / §2 Static Generator."""
    Mlen = len(z_samples)
    Cfull = np.fft.fft(z_samples) / Mlen
    C = np.zeros(2 * N + 1, dtype=complex)
    for idx, k in enumerate(range(-N, N + 1)):
        C[idx] = Cfull[k % Mlen]
    return C


def reconstruct(C, t):
    """z(t) = sum_k C_k * e^{i 2pi k t} , spec §1 inverse transform."""
    ks = np.arange(-N, N + 1)
    mat = np.exp(1j * 2 * np.pi * np.outer(t, ks))
    return mat @ C


PRECOMPUTED = {name: curve_to_coeffs(fn()) for name, fn in SHAPES.items()}

# All four generators are symmetric about the origin, so C0 = 0 for each of
# them by construction. That's the degenerate case, not the realistic one:
# a shape actually drawn on a canvas is essentially never centered exactly
# on the origin. Give every precomputed shape a fixed, nonzero baseline C0
# ("canvas position") so the DC-boost probe (Z/X) has something to act on.
_CANVAS_OFFSET = 0.35 + 0.15j
for _name in PRECOMPUTED:
    PRECOMPUTED[_name][k_index(0)] += _CANVAS_OFFSET

# ---------------------------------------------------------------------------
# §4, Operators
# ---------------------------------------------------------------------------

def op_phase_shift(C, theta):
    """R_theta: diagonal unitary, e^{i theta} applied uniformly. §4B."""
    return C * np.exp(1j * theta)


def op_freq_shift(C, m):
    """H_m: C_out[k] = C_in[k - m]. Linear (non-wrapping) shift, so any
    energy pushed past +-N is truncated, i.e. lost. §4C."""
    Cout = np.zeros_like(C)
    for idx, k in enumerate(range(-N, N + 1)):
        src_k = k - m
        if -N <= src_k <= N:
            Cout[idx] = C[k_index(src_k)]
    return Cout


def op_spectral_filter(C, gain, mode, cutoff):
    """W: real non-negative diagonal filter. Uniform gain is the same
    operator family with w_k = gain for all k. §4D."""
    w = np.full(2 * N + 1, gain, dtype=float)
    if mode == "low":
        for idx, k in enumerate(range(-N, N + 1)):
            if abs(k) > cutoff:
                w[idx] = 0.0
    elif mode == "high":
        for idx, k in enumerate(range(-N, N + 1)):
            if abs(k) <= cutoff:
                w[idx] = 0.0
    return C * w


def op_dc_boost(C, boost):
    """Test probe only (not in spec as-is): isolates C0 to demonstrate
    that it sits outside the 'shape' subspace eta measures."""
    Cout = C.copy()
    Cout[k_index(0)] *= boost
    return Cout


def op_attenuate_noise(C, d, noise_std):
    """§5.2: exponential high-frequency damping + additive complex
    thermal noise."""
    ks = np.arange(-N, N + 1)
    atten = np.exp(-ALPHA * np.abs(ks) * d)
    Cout = C * atten
    if noise_std > 0:
        noise = (np.random.normal(0, noise_std, size=Cout.shape)
                 + 1j * np.random.normal(0, noise_std, size=Cout.shape))
        Cout = Cout + noise
    return Cout


def op_power_clamp(C):
    """§5.1: scale down to the power ceiling if exceeded."""
    p = float(np.sum(np.abs(C) ** 2))
    if p > P_MAX:
        C = C * math.sqrt(P_MAX / p)
    return C


def run_pipeline(source_C, st):
    steps = {}
    C = source_C.copy()
    steps["source"] = C.copy()

    C = op_phase_shift(C, st["theta"])
    steps["after_phase"] = C.copy()

    C = op_freq_shift(C, st["m"])
    steps["after_freqshift"] = C.copy()

    C = op_spectral_filter(C, st["gain"], st["filter_mode"], st["cutoff"])
    steps["after_filter"] = C.copy()

    C = op_dc_boost(C, st["dc_boost"])
    steps["after_dcboost"] = C.copy()

    C = op_attenuate_noise(C, st["distance"], NOISE_STD if st["noise_on"] else 0.0)
    steps["after_atten"] = C.copy()

    if st["clamp_on"]:
        C = op_power_clamp(C)
    steps["final"] = C.copy()

    return C, steps

# ---------------------------------------------------------------------------
# §3, RX
# ---------------------------------------------------------------------------

def norms_and_eta(Cin, T):
    idx0 = k_index(0)
    mask = np.ones(2 * N + 1, dtype=bool)
    mask[idx0] = False
    Cin_shape, T_shape = Cin[mask], T[mask]

    norm2_full_in = float(np.sum(np.abs(Cin) ** 2))
    norm2_shape_in = float(np.sum(np.abs(Cin_shape) ** 2))
    norm2_shape_T = float(np.sum(np.abs(T_shape) ** 2))

    inner_shape = np.sum(Cin_shape * np.conj(T_shape))
    denom = math.sqrt(norm2_shape_in) * math.sqrt(norm2_shape_T)
    eta = abs(inner_shape) / denom if denom > 1e-12 else 0.0
    return norm2_full_in, norm2_shape_in, norm2_shape_T, eta


def y_rx(norm2_full_in, norm2_shape_in, eta):
    if eta < ETA_THRESH:
        return 0.0, 0.0
    spec_formula = GAMMA * norm2_full_in * eta     # literal spec §3.4 (uses FULL norm)
    fixed_formula = GAMMA * norm2_shape_in * eta   # corrected (shape-only norm)
    return spec_formula, fixed_formula

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def to_screen(z, cx, cy, scale):
    return (int(cx + z.real * scale), int(cy - z.imag * scale))


def draw_curve(surf, C, cx, cy, scale, color, width=2):
    t = np.linspace(0, 1, M, endpoint=False)
    pts = reconstruct(C, t)
    xy = [to_screen(z, cx, cy, scale) for z in pts]
    if len(xy) > 1:
        pygame.draw.aalines(surf, color, True, xy)


def draw_epicycles(surf, C, cx, cy, scale, t_now, color):
    order = sorted(range(2 * N + 1), key=lambda i: -abs(C[i]))
    x, y = cx, cy
    for idx in order:
        k = idx - N
        r = abs(C[idx]) * scale
        prev = (x, y)
        cval = C[idx] * np.exp(1j * 2 * np.pi * k * t_now)
        x = x + cval.real * scale
        y = y - cval.imag * scale
        if r > 0.6:
            pygame.draw.circle(surf, DIM, prev, max(int(r), 1), 1)
        pygame.draw.line(surf, GREY, prev, (int(x), int(y)), 1)
    pygame.draw.circle(surf, YELLOW, (int(x), int(y)), 4)


def text_lines(font, lines, x, y, surf, color=WHITE, dy=20):
    for i, line in enumerate(lines):
        c = color[i] if isinstance(color, list) else color
        surf.blit(font.render(line, True, c), (x, y + i * dy))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fourier Signal Engine, Test Bench")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    font_small = pygame.font.SysFont("consolas", 14)

    default_state = dict(
        source="star", target="star",
        theta=0.0, m=0, gain=1.0,
        filter_mode="none", cutoff=N,
        dc_boost=1.0,
        distance=0.0, noise_on=False, clamp_on=False,
    )
    st = default_state.copy()
    t_now = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t_now = (t_now + dt * 0.15) % 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    st = default_state.copy()
                elif event.key in SOURCE_KEYMAP:
                    st["source"] = SOURCE_KEYMAP[event.key]
                elif event.key in TARGET_KEYMAP:
                    st["target"] = TARGET_KEYMAP[event.key]
                elif event.key == pygame.K_f:
                    st["filter_mode"] = {"none": "low", "low": "high",
                                          "high": "none"}[st["filter_mode"]]
                elif event.key == pygame.K_n:
                    st["noise_on"] = not st["noise_on"]
                elif event.key == pygame.K_c:
                    st["clamp_on"] = not st["clamp_on"]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            st["theta"] -= 0.03
        if keys[pygame.K_RIGHT]:
            st["theta"] += 0.03
        if keys[pygame.K_UP]:
            st["gain"] = min(st["gain"] + 0.02, 5.0)
        if keys[pygame.K_DOWN]:
            st["gain"] = max(st["gain"] - 0.02, 0.0)
        if keys[pygame.K_LEFTBRACKET]:
            st["m"] -= 1
        if keys[pygame.K_RIGHTBRACKET]:
            st["m"] += 1
        if keys[pygame.K_MINUS]:
            st["cutoff"] = max(st["cutoff"] - 1, 0)
        if keys[pygame.K_EQUALS]:
            st["cutoff"] = min(st["cutoff"] + 1, N)
        if keys[pygame.K_z]:
            st["dc_boost"] = max(st["dc_boost"] - 0.05, 0.0)
        if keys[pygame.K_x]:
            st["dc_boost"] = min(st["dc_boost"] + 0.05, 10.0)
        if keys[pygame.K_COMMA]:
            st["distance"] = max(st["distance"] - 0.02, 0.0)
        if keys[pygame.K_PERIOD]:
            st["distance"] = min(st["distance"] + 0.02, 20.0)
        if keys[pygame.K_LEFTBRACKET] or keys[pygame.K_RIGHTBRACKET]:
            st["m"] = int(np.clip(st["m"], -2 * N, 2 * N))

        # avoid re-triggering the bracket keys every frame at full speed
        pygame.time.wait(0)

        source_C = PRECOMPUTED[st["source"]]
        target_C = PRECOMPUTED[st["target"]]
        S_in, steps = run_pipeline(source_C, st)

        n_full, n_shape, n_shape_T, eta = norms_and_eta(S_in, target_C)
        y_spec, y_fixed = y_rx(n_full, n_shape, eta)

        # energy audit around H_m (phase shift preserves norm, so any
        # drop between after_phase and after_freqshift is truncation loss)
        e_before = float(np.sum(np.abs(steps["after_phase"]) ** 2))
        e_after = float(np.sum(np.abs(steps["after_freqshift"]) ** 2))
        energy_lost = max(e_before - e_after, 0.0)

        # ---------------- draw ----------------
        screen.fill(BG)

        cx1, cy1 = 90 + PANEL // 2, 70 + PANEL // 2
        scale = PANEL / 2.6
        pygame.draw.rect(screen, (26, 26, 32), (70, 50, PANEL + 40, PANEL + 40), border_radius=8)
        draw_curve(screen, target_C, cx1, cy1, scale, GREY, width=1)
        draw_curve(screen, S_in, cx1, cy1, scale, CYAN if eta >= ETA_THRESH else RED)
        draw_epicycles(screen, S_in, cx1, cy1, scale, t_now, CYAN)
        text_lines(font_small, ["grey = target T", "cyan/red = S_in (pass/fail eta)"],
                   80, 55, screen, GREY, dy=16)

        hud_x = 90 + PANEL + 70
        lines = [
            f"SOURCE: {st['source']:<8}  TARGET: {st['target']:<8}",
            "",
            f"theta (phase):   {st['theta']:+.2f} rad        [LEFT/RIGHT]",
            f"gain (uniform):  {st['gain']:.2f}               [UP/DOWN]",
            f"freq shift m:    {st['m']:+d}                  [ [ / ] ]",
            f"filter mode:     {st['filter_mode']:<5} cutoff K={st['cutoff']}   [F]  [-/=]",
            f"DC boost |C0|:   {st['dc_boost']:.2f}               [Z/X]  <-- exploit probe",
            f"distance d:      {st['distance']:.2f}               [, / .]",
            f"noise:           {'ON' if st['noise_on'] else 'off'}                   [N]",
            f"power clamp:     {'ON' if st['clamp_on'] else 'off'} (P_max={P_MAX})   [C]",
            "",
            "---------------- RX (spec §3) ----------------",
            f"||S_in||^2 (full, incl. C0):   {n_full:8.3f}",
            f"||S_in||^2 (shape, excl. C0):  {n_shape:8.3f}",
            f"||T||^2 (shape, excl. C0):     {n_shape_T:8.3f}",
            f"eta (shape match):             {eta:8.4f}   {'PASS' if eta>=ETA_THRESH else 'fail'} (thresh {ETA_THRESH})",
            "",
            f"y_RX  [spec formula, full norm]:   {y_spec:9.3f}",
            f"y_RX  [fixed formula, shape norm]: {y_fixed:9.3f}",
            "  ^ raise DC boost (X) and watch: eta + fixed formula stay",
            "    flat while the spec formula climbs -> the C0 exploit.",
            "",
            "---------------- H_m energy audit (§4C / §5) ----------------",
            f"||S|| ^2 before shift: {e_before:8.3f}",
            f"||S|| ^2 after shift:  {e_after:8.3f}",
            f"energy lost to truncation: {energy_lost:8.3f}",
            "  ^ nonzero whenever a harmonic is pushed past +-N.",
        ]
        text_lines(font, lines, hud_x, 60, screen, WHITE, dy=20)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
