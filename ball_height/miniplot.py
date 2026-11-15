"""Tiny PIL-based plotting (matplotlib is not available in this environment)."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

COLORS = dict(blue=(31, 119, 180), orange=(255, 127, 14), green=(44, 160, 44), red=(214, 39, 40), purple=(148, 103, 189),
              gray=(127, 127, 127), black=(0, 0, 0), brown=(140, 86, 75), pink=(227, 119, 194), cyan=(23, 190, 207),
              lightgray=(210, 210, 210), gold=(230, 180, 0))


def font(size=14):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def nice_ticks(lo, hi, n=8):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * mag - raw)) * mag
    t0 = np.ceil(lo / step) * step
    return list(np.arange(t0, hi + 1e-9, step))


class Plot:
    def __init__(self, w=1600, h=500, xlim=(0, 1), ylim=(0, 1), title='', xlabel='', ylabel='', margin=(70, 20, 40, 45)):
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = margin
        self.xlim, self.ylim = xlim, ylim
        self.im = Image.new('RGB', (w, h), 'white')
        self.d = ImageDraw.Draw(self.im)
        self.f = font(13); self.fb = font(16)
        d = self.d
        for xt in nice_ticks(*xlim, n=14):
            x = self.X(xt)
            d.line([(x, self.mt), (x, h - self.mb)], fill=(232, 232, 232))
            d.text((x, h - self.mb + 4), ('%g' % xt), fill='black', font=self.f, anchor='ma')
        for yt in nice_ticks(*ylim, n=8):
            y = self.Y(yt)
            d.line([(self.ml, y), (w - self.mr, y)], fill=(232, 232, 232))
            d.text((self.ml - 6, y), ('%g' % yt), fill='black', font=self.f, anchor='rm')
        d.rectangle([self.ml, self.mt, w - self.mr, h - self.mb], outline='black')
        d.text((w // 2, 6), title, fill='black', font=self.fb, anchor='ma')
        d.text((w // 2, h - 16), xlabel, fill='black', font=self.f, anchor='ma')
        d.text((6, self.mt - 18), ylabel, fill='black', font=self.f)
        self._leg = []

    def X(self, x):
        return self.ml + (np.asarray(x, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.w - self.ml - self.mr)

    def Y(self, y):
        return self.h - self.mb - (np.asarray(y, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.h - self.mt - self.mb)

    def line(self, x, y, color='blue', width=2, label=None):
        c = COLORS.get(color, color)
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        seg = []
        for i in range(len(x)):
            if ok[i]:
                seg.append((float(self.X(x[i])), float(self.Y(y[i]))))
            else:
                if len(seg) > 1:
                    self.d.line(seg, fill=c, width=width)
                seg = []
        if len(seg) > 1:
            self.d.line(seg, fill=c, width=width)
        if label:
            self._leg.append((label, c, 'line'))

    def scatter(self, x, y, color='red', r=3, label=None, fill=True):
        c = COLORS.get(color, color)
        for xi, yi in zip(np.asarray(x, float), np.asarray(y, float)):
            if np.isfinite(xi) and np.isfinite(yi):
                X, Y = float(self.X(xi)), float(self.Y(yi))
                if fill:
                    self.d.ellipse([X - r, Y - r, X + r, Y + r], fill=c)
                else:
                    self.d.ellipse([X - r, Y - r, X + r, Y + r], outline=c)
        if label:
            self._leg.append((label, c, 'dot'))

    def vline(self, x, color='gray', width=1, dash=False):
        c = COLORS.get(color, color)
        X = float(self.X(x))
        if dash:
            for y in range(self.mt, self.h - self.mb, 8):
                self.d.line([(X, y), (X, y + 4)], fill=c, width=width)
        else:
            self.d.line([(X, self.mt), (X, self.h - self.mb)], fill=c, width=width)

    def hline(self, y, color='gray', width=1):
        c = COLORS.get(color, color)
        Y = float(self.Y(y))
        self.d.line([(self.ml, Y), (self.w - self.mr, Y)], fill=c, width=width)

    def band(self, x0, x1, color=(240, 240, 200)):
        self.d.rectangle([float(self.X(x0)), self.mt + 1, float(self.X(x1)), self.h - self.mb - 1], fill=color)

    def text(self, x, y, s, color='black', anchor='la'):
        self.d.text((float(self.X(x)), float(self.Y(y))), s, fill=COLORS.get(color, color), font=self.f, anchor=anchor)

    def legend(self, x=None, y=None):
        x = self.ml + 10 if x is None else x
        y = self.mt + 8 if y is None else y
        for lab, c, kind in self._leg:
            if kind == 'line':
                self.d.line([(x, y + 8), (x + 24, y + 8)], fill=c, width=3)
            else:
                self.d.ellipse([x + 8, y + 4, x + 16, y + 12], fill=c)
            self.d.text((x + 30, y), lab, fill='black', font=self.f)
            y += 18

    def save(self, path):
        self.im.save(path)


def stack(images, path):
    w = max(i.w for i in images)
    h = sum(i.h for i in images)
    out = Image.new('RGB', (w, h), 'white')
    y = 0
    for i in images:
        out.paste(i.im, (0, y)); y += i.h
    out.save(path)
