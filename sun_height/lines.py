# sample the blue court lines along a frame; prints JSON {line: [[x,y],...]}
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, json, numpy as np
from PIL import Image
im = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(float)
blue = im[...,2] - (im[...,0]+im[...,1])/2
k = np.ones(3)/3
def peak(p):
    p = np.convolve(p, k, "same"); p = p - np.median(p); i = int(np.argmax(p))
    if p[i] < 15 or i < 3 or i > len(p)-4: return None
    a,b,c = p[i-1],p[i],p[i+1]; den = a-2*b+c
    return i + (0.5*(a-c)/den if den else 0)
def track(start, step_axis, n, step, win):
    """follow a line: step along step_axis, search +-win across"""
    pts = []; x, y = start
    for _ in range(n):
        if step_axis == 'x':
            if not 0 <= x < im.shape[1]: break
            y0 = int(round(y))-win; p = peak(blue[max(y0,0):y0+2*win+1, int(x)])
            if p is not None: y = y0+p; pts.append([x, y])
            x += step
        else:
            if not 0 <= y < im.shape[0]: break
            x0 = int(round(x))-win; p = peak(blue[int(y), max(x0,0):x0+2*win+1])
            if p is not None: x = x0+p; pts.append([x, y])
            y += step
    return pts
out = {
 'far':   track((232, 419), 'x', 156, 10, 6),
 'near':  track((50, 960), 'x', 190, 10, 6),
 'left':  track((213.5, 440), 'y', 70, 3, 8),
 'right': track((1832.5, 460), 'y', 30, 3, 8),
}
print(json.dumps(out))
