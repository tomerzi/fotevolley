"""Helper: draw a labelled pixel grid on (a crop of) a screenshot for reading coordinates by eye."""
import sys, cv2
src, out = sys.argv[1], sys.argv[2]
x0, y0, x1, y1, s, step = (float(v) for v in sys.argv[3:9])
x0, y0, x1, y1, step = int(x0), int(y0), int(x1), int(y1), int(step)
im = cv2.imread(src)
x1, y1 = min(x1, im.shape[1]), min(y1, im.shape[0])
c = cv2.resize(im[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
for x in range(x0 - x0 % step + step, x1, step):
    big = x % (step * 5) == 0
    cv2.line(c, (int((x - x0) * s), 0), (int((x - x0) * s), c.shape[0]), (0, 0, 255) if big else (0, 255, 255), 1)
    if big: cv2.putText(c, str(x), (int((x - x0) * s) + 2, 14), 0, 0.45, (0, 0, 255), 1)
for y in range(y0 - y0 % step + step, y1, step):
    big = y % (step * 5) == 0
    cv2.line(c, (0, int((y - y0) * s)), (c.shape[1], int((y - y0) * s)), (0, 0, 255) if big else (0, 255, 255), 1)
    if big: cv2.putText(c, str(y), (2, int((y - y0) * s) - 2), 0, 0.45, (255, 0, 0), 1)
cv2.imwrite(out, c)
