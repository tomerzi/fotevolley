"""Sub-pixel extraction of the blue sidelines (blue ridge) and the net top tape upper edge,
starting from rough endpoints read by eye. Usage from other scripts: ridge(img, p0, p1, kind)."""
import numpy as np, cv2


def ridge(im, p0, p1, kind='blue', half=6, step=3, min_rel=0.6):
    im = im.astype(np.float32)
    B, G, R = cv2.split(im)
    if kind == 'blue':
        sig = cv2.GaussianBlur(B - R, (0, 0), 1.2)
    else:  # tape upper edge: dark above, bright below -> vertical gradient
        g = cv2.GaussianBlur(im.mean(2), (0, 0), 1.0)
        sig = np.zeros_like(g); sig[1:] = g[1:] - g[:-1]
    p0, p1 = np.array(p0, float), np.array(p1, float)
    steep = abs(p1[1] - p0[1]) > abs(p1[0] - p0[0])
    pts = []
    for _ in range(3):
        pts = []
        if steep:
            a = (p1[0] - p0[0]) / (p1[1] - p0[1]); b = p0[0] - a * p0[1]
            for y in np.arange(min(p0[1], p1[1]), max(p0[1], p1[1]), step):
                y = int(y); xe = a * y + b
                xs = np.arange(int(xe - half), int(xe + half + 1))
                xs = xs[(xs >= 0) & (xs < im.shape[1])]
                if len(xs) < 3 or y >= im.shape[0]: continue
                v = sig[y, xs]; k = int(np.argmax(v))
                if 0 < k < len(xs) - 1:
                    den = v[k - 1] - 2 * v[k] + v[k + 1]
                    d = (v[k - 1] - v[k + 1]) / (2 * den) if den != 0 else 0
                    pts.append((xs[k] + d, y, v[k] - np.median(v)))
        else:
            a = (p1[1] - p0[1]) / (p1[0] - p0[0]); b = p0[1] - a * p0[0]
            for x in np.arange(min(p0[0], p1[0]), max(p0[0], p1[0]), step):
                x = int(x); ye = a * x + b
                ys = np.arange(int(ye - half), int(ye + half + 1))
                ys = ys[(ys >= 0) & (ys < im.shape[0])]
                if len(ys) < 3 or x >= im.shape[1]: continue
                v = sig[ys, x]; k = int(np.argmax(v))
                if 0 < k < len(ys) - 1:
                    den = v[k - 1] - 2 * v[k] + v[k + 1]
                    d = (v[k - 1] - v[k + 1]) / (2 * den) if den != 0 else 0
                    pts.append((x, ys[k] + d - (0.5 if kind != 'blue' else 0), v[k] - np.median(v)))
        P = np.array(pts)
        m = P[:, 2] > P[:, 2].mean() * min_rel
        P = P[m]
        # robust line refit, drop >2px outliers
        for _ in range(2):
            if steep:
                c = np.polyfit(P[:, 1], P[:, 0], 1); r = P[:, 0] - np.polyval(c, P[:, 1])
            else:
                c = np.polyfit(P[:, 0], P[:, 1], 1); r = P[:, 1] - np.polyval(c, P[:, 0])
            P = P[np.abs(r) < max(2.0, 2.5 * r.std())]
        lo, hi = (P[:, 1].min(), P[:, 1].max()) if steep else (P[:, 0].min(), P[:, 0].max())
        if steep:
            p0, p1 = np.array([np.polyval(c, lo), lo]), np.array([np.polyval(c, hi), hi])
        else:
            p0, p1 = np.array([lo, np.polyval(c, lo)]), np.array([hi, np.polyval(c, hi)])
    return P[:, :2], float(r.std())
