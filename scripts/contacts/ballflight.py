"""Split the ball path into free-flight pieces; each break between two pieces is a touch (or bounce).

Between touches the ball only feels gravity (and a little drag), so over one flight its image x(t) and y(t)
are close to quadratics. A dynamic programme finds the fewest quadratic pieces that explain the observed
ball centres within the detection noise. Each break is an event, located where the two neighbouring pieces
meet, which also finds touches hidden in a short detection gap (ball behind a body)."""
import numpy as np

SIGMA = 2.0          # px, ball-centre noise
HUBER = 3.0          # residuals beyond HUBER*SIGMA count linearly (one bad detection must not split a flight)
SEG_PEN = 40.0       # cost of one extra piece (in squared-sigma units)
MAX_LEN = 90         # frames in one piece at most
MIN_PTS = 3
RAT_MIN = 9           # points needed for the perspective (rational) model


def _huber(r2):
    c2 = HUBER ** 2
    return np.where(r2 <= c2, r2, 2 * HUBER * np.sqrt(r2) - c2)


def _solve(s, x, y):
    """free flight seen by a pinhole camera: the 3D point is quadratic in time, so in the image
    x = (a0 + a1 s + a2 s^2) / (1 + d1 s + d2 s^2), y = (b0 + b1 s + b2 s^2) / (same).
    Short pieces use a plain quadratic (the denominator is ~1 over a few frames)."""
    n = len(s)
    if n < RAT_MIN:
        deg = 2 if n >= 4 else 1
        return np.polyfit(s, x, deg), np.polyfit(s, y, deg), np.zeros(2)
    z, o = np.zeros((n, 3)), np.ones(n)
    V = np.c_[o, s, s * s]
    A = np.r_[np.c_[V, z, -x * s, -x * s * s], np.c_[z, V, -y * s, -y * s * s]]
    p, *_ = np.linalg.lstsq(A, np.r_[x, y], rcond=None)
    d = p[6:8]
    ss = np.linspace(-3, s[-1] + 3, 50)
    w = 1 + d[0] * ss + d[1] * ss * ss
    if not np.all(np.isfinite(p)) or w.min() < 0.5 or w.max() > 2:   # no pole/huge depth change near the piece
        return np.polyfit(s, x, 2), np.polyfit(s, y, 2), np.zeros(2)
    return p[2::-1], p[5:2:-1], d


def _eval(m, s):
    px, py, d = m
    w = 1 + d[0] * s + d[1] * s * s
    return np.polyval(px, s) / w, np.polyval(py, s) / w


def _fit(t, x, y):
    t0 = t[0]
    s = (t - t0).astype(float)
    m = _solve(s, x, y)
    ex, ey = _eval(m, s)
    r2 = ((ex - x) ** 2 + (ey - y) ** 2) / SIGMA ** 2
    # one refit without the worst point if it is far off (a single wrong detection)
    if len(t) >= 6 and r2.max() > 25:
        k = np.ones(len(t), bool); k[np.argmax(r2)] = False
        m = _solve(s[k], x[k], y[k])
        ex, ey = _eval(m, s)
        r2 = ((ex - x) ** 2 + (ey - y) ** 2) / SIGMA ** 2
    return float(_huber(r2).sum()), (t0,) + m


class Piece:
    def __init__(self, t, model):
        self.t_first, self.t_last = int(t[0]), int(t[-1])
        self.n = len(t)
        self.t0, self.m = model[0], model[1:]

    def pos(self, t):
        return np.array(_eval(self.m, np.float64(t - self.t0)))

    def vel(self, t, h=0.25):
        return (self.pos(t + h) - self.pos(t - h)) / (2 * h)


def segment(t, x, y):
    """t: increasing frame numbers of observed ball centres (one camera shot). Returns a list of Pieces."""
    n = len(t)
    if n < MIN_PTS:
        return []
    best = np.full(n + 1, np.inf); best[0] = 0.0
    back = np.zeros(n + 1, int); model = [None] * (n + 1)
    for j in range(1, n + 1):                 # piece covers points i..j-1
        for i in range(j - MIN_PTS, -1, -1):
            if t[j - 1] - t[i] > MAX_LEN:
                break
            if not np.isfinite(best[i]):
                continue
            # a gap longer than 12 frames inside one piece is not trusted
            if i < j - 1 and np.max(np.diff(t[i:j])) > 12:
                break
            c, m = _fit(t[i:j], x[i:j], y[i:j])
            if not np.isfinite(c):
                continue
            c += best[i] + SEG_PEN
            if c < best[j]:
                best[j], back[j], model[j] = c, i, m
        # leftover single points may be skipped (outliers) at a cost
        if best[j - 1] + 12 < best[j]:
            best[j], back[j], model[j] = best[j - 1] + 12, j - 1, None
    pieces, j = [], n
    while j > 0:
        i = back[j]
        if model[j] is not None:
            pieces.append(Piece(t[i:j], model[j]))
        j = i
    return pieces[::-1]


def events(pieces):
    """break between consecutive pieces -> dict(frame, pos, v_in, v_out, gap)."""
    out = []
    for a, b in zip(pieces[:-1], pieces[1:]):
        lo, hi = a.t_last, b.t_first
        ts = np.arange(lo, hi + 1, 0.25)
        d = [np.hypot(*(a.pos(s) - b.pos(s))) for s in ts]
        k = int(np.argmin(d)); s = ts[k]
        f = int(round(s))
        out.append(dict(frame=f, t_exact=float(s), pos=(a.pos(s) + b.pos(s)) / 2, v_in=a.vel(s), v_out=b.vel(s),
                        miss_px=float(d[k]), gap=hi - lo, n_in=a.n, n_out=b.n))
    return out
