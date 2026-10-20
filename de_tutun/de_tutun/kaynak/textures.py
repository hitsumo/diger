"""de_tutun için yordamsal dokular -> tutun.wad (WAD3).

Terk edilmiş tütün fabrikası: isli tuğla, grafiti, çatlak sıva ve beton, paslı sac,
kırık pencere, tahta perde, barut fıçısı, tütün sandığı, çöp yığını, tel örgü.

Kullanım: python textures.py [onizleme_dizini]
"""
import os
import struct
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(1923)
FONT_DIR = "/usr/share/fonts/truetype"
FONTS = {
    "bold": f"{FONT_DIR}/dejavu/DejaVuSans-Bold.ttf",
    "oblique": f"{FONT_DIR}/freefont/FreeSansBoldOblique.ttf",
    "serif": f"{FONT_DIR}/dejavu/DejaVuSerif-Bold.ttf",
    "mono": f"{FONT_DIR}/dejavu/DejaVuSansMono-Bold.ttf",
}


# ---------------------------------------------------------------- yardımcılar
def noise(w, h, sigma, amp=1.0):
    n = rng.standard_normal((h, w))
    n = gaussian_filter(n, sigma, mode="wrap")
    n /= (np.abs(n).max() + 1e-9)
    return n * amp


def fbm(w, h, amp=1.0):
    return (noise(w, h, 0.6) * 0.35 + noise(w, h, 2) * 0.35 + noise(w, h, 6) * 0.3) * amp


def solid(w, h, rgb):
    a = np.zeros((h, w, 3), float)
    a[:] = rgb
    return a


def add_noise(a, amp, per_channel=False):
    h, w = a.shape[:2]
    if per_channel:
        for c in range(3):
            a[..., c] += fbm(w, h, amp)
    else:
        a += fbm(w, h, amp)[..., None]
    return a


def rect(a, x0, y0, x1, y1, rgb, alpha=1.0):
    a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - alpha) + np.array(rgb) * alpha


def stains(a, count, rmin, rmax, rgb, alpha=0.5, sigma=3):
    """Yumuşak kenarlı lekeler (is, nem, pas)."""
    h, w = a.shape[:2]
    m = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(count):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(rmin, rmax)
        for ox in (-w, 0, w):
            for oy in (-h, 0, h):
                m = np.maximum(m, np.clip(1 - np.hypot(xx - cx + ox, yy - cy + oy) / r, 0, 1))
    m = gaussian_filter(m, sigma, mode="wrap")
    m = np.clip(m * 1.4, 0, 1) * alpha
    a[:] = a * (1 - m[..., None]) + np.array(rgb) * m[..., None]
    return a


def cracks(a, count, rgb, length=60, width=1):
    """Rastgele yürüyüşle çatlak çizgileri."""
    h, w = a.shape[:2]
    for _ in range(count):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(length):
            ang += rng.normal(0, 0.35)
            x += np.cos(ang); y += np.sin(ang)
            xi, yi = int(x) % w, int(y) % h
            a[yi:yi + width, xi:xi + width] = rgb
    return a


def blocks(w, h, bw, bh, base, mortar, var=12, mortar_w=2, bevel=0, stagger=True):
    a = solid(w, h, base)
    rows = h // bh
    for r in range(rows):
        off = (bw // 2) if (stagger and r % 2) else 0
        for c in range(-1, w // bw + 1):
            x0 = c * bw + off
            shade = rng.uniform(-var, var)
            xs = np.arange(x0, x0 + bw) % w
            a[r * bh:(r + 1) * bh][:, xs] += shade
            if bevel:
                a[r * bh:r * bh + bevel][:, xs] += 14
                a[(r + 1) * bh - bevel:(r + 1) * bh][:, xs] -= 18
            a[r * bh:(r + 1) * bh][:, xs[:mortar_w]] = mortar
        a[r * bh:r * bh + mortar_w] = mortar
    return add_noise(a, 10)


def text_layer(w, h, text, font, size, fill, outline, pos=None, angle=0.0, shadow=None):
    """Şeffaf katmana yazı: dış çizgi + dolgu; döndürülebilir. (RGBA float döner)"""
    f = ImageFont.truetype(FONTS[font], size)
    pad = size
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    if pos is None:
        pos = ((w - tw) // 2, (h - th) // 2)
    x, y = pos[0] + pad - bb[0], pos[1] + pad - bb[1]
    if shadow:
        d.text((x + 3, y + 3), text, font=f, fill=shadow + (255,))
    d.text((x, y), text, font=f, fill=fill + (255,), stroke_width=max(2, size // 10), stroke_fill=outline + (255,))
    if angle:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, center=(x + tw / 2, y + th / 2))
    img = img.crop((pad, pad, pad + w, pad + h))
    return np.asarray(img).astype(float)


def paste_rgba(a, layer, alpha=1.0):
    al = layer[..., 3:4] / 255.0 * alpha
    a[:] = a * (1 - al) + layer[..., :3] * al
    return a


def drips(a, layer, rgb, count=12, maxlen=40):
    """Yazının alt kenarından aşağı akan boya damlaları."""
    h, w = a.shape[:2]
    mask = layer[..., 3] > 128
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return a
    for _ in range(count):
        x = int(rng.choice(cols))
        ys = np.nonzero(mask[:, x])[0]
        y0 = ys.max()
        ln = int(rng.uniform(6, maxlen))
        wd = int(rng.integers(1, 3))
        a[y0:min(h, y0 + ln), x:x + wd] = a[y0:min(h, y0 + ln), x:x + wd] * 0.3 + np.array(rgb) * 0.7
    return a


# ---------------------------------------------------------------- duvarlar
def brick(sooty=0.35):
    a = blocks(128, 128, 32, 16, (104, 58, 44), (82, 76, 68), var=16, mortar_w=2, bevel=1)
    stains(a, 5, 14, 40, (34, 28, 26), alpha=sooty, sigma=4)
    stains(a, 3, 8, 20, (120, 92, 70), alpha=0.25, sigma=3)
    return a


def brick_dark():
    a = blocks(128, 128, 32, 16, (74, 46, 38), (58, 54, 50), var=14, mortar_w=2, bevel=1)
    stains(a, 7, 16, 48, (26, 22, 22), alpha=0.55, sigma=4)
    return a


def plaster():
    a = solid(128, 128, (146, 138, 118))
    add_noise(a, 10)
    stains(a, 6, 12, 40, (92, 84, 70), alpha=0.45, sigma=4)
    stains(a, 3, 6, 16, (60, 58, 56), alpha=0.35, sigma=2)
    cracks(a, 4, (70, 64, 56), length=70)
    # dökülmüş sıva: tuğla görünen yama
    yy, xx = np.mgrid[0:128, 0:128]
    cx, cy = rng.uniform(20, 108), rng.uniform(20, 108)
    m = (np.hypot(xx - cx, (yy - cy) * 1.4) + noise(128, 128, 3, 10)) < 22
    b = blocks(128, 128, 32, 16, (104, 58, 44), (82, 76, 68), var=12)
    a[m] = b[m]
    return a


def graffiti(base, texts, w=256, h=128):
    """Taban duvar dokusu (w x h) üstüne birkaç renkli tag."""
    a = base
    for text, font, size, fill, outline, pos, ang in texts:
        layer = text_layer(w, h, text, font, size, fill, outline, pos, ang)
        paste_rgba(a, layer, 0.92)
        drips(a, layer, fill, count=8, maxlen=28)
    return a


def wall_wide(fn):
    """128x128 duvar dokusunu 256x128'e döşer (grafiti tabanı)."""
    t = fn()
    return np.concatenate([t, np.roll(t, 7, axis=0)], axis=1)


def graf1():
    base = wall_wide(brick)
    return graffiti(base, [
        ("DUMAN", "oblique", 56, (230, 60, 40), (30, 10, 10), (18, 24), 4),
        ("1923", "bold", 30, (240, 220, 60), (20, 20, 20), (170, 80), -6),
    ])


def graf2():
    base = wall_wide(plaster)
    return graffiti(base, [
        ("GÖLGE", "bold", 50, (60, 200, 110), (10, 40, 20), (24, 20), -3),
        ("kaçak", "oblique", 34, (250, 250, 250), (40, 40, 40), (140, 76), 8),
    ])


def graf3():
    base = wall_wide(brick)
    return graffiti(base, [
        ("RUHSUZ", "oblique", 48, (120, 90, 230), (20, 10, 50), (12, 34), 2),
        ("x", "bold", 40, (240, 120, 40), (30, 10, 0), (205, 12), 15),
    ])


def graf4():
    base = wall_wide(lambda: concrete_wall())
    return graffiti(base, [
        ("SESSİZ OL", "bold", 34, (240, 240, 240), (20, 20, 20), (10, 36), -2),
        ("!", "oblique", 70, (230, 40, 40), (40, 0, 0), (222, 20), 0),
    ])


def concrete_wall():
    a = solid(128, 128, (128, 126, 120))
    add_noise(a, 8)
    for y in (0, 64):
        a[y:y + 2] -= 24          # kalıp derzleri
    a[:, 0:2] -= 24
    stains(a, 5, 10, 36, (78, 76, 72), alpha=0.4, sigma=4)
    stains(a, 2, 8, 24, (96, 70, 52), alpha=0.3, sigma=3)    # pas akıntısı
    cracks(a, 3, (74, 72, 68), length=50)
    return a


def concrete_floor():
    a = solid(128, 128, (112, 110, 104))
    add_noise(a, 10)
    stains(a, 7, 10, 44, (66, 64, 60), alpha=0.45, sigma=5)
    cracks(a, 6, (60, 58, 54), length=60)
    return a


def asphalt():
    a = solid(128, 128, (58, 58, 60))
    add_noise(a, 9)
    a += (noise(128, 128, 0.5) * 9)[..., None]
    stains(a, 4, 12, 40, (36, 36, 40), alpha=0.5, sigma=5)     # yağ lekeleri
    cracks(a, 5, (40, 40, 42), length=70)
    return a


def dirt():
    a = solid(128, 128, (78, 64, 48))
    add_noise(a, 12)
    a += (noise(128, 128, 0.7) * 10)[..., None]
    stains(a, 5, 10, 30, (54, 44, 34), alpha=0.5, sigma=4)
    return a


def gravel():
    a = solid(128, 128, (92, 88, 80))
    a += (noise(128, 128, 0.6) * 22)[..., None]
    a += (noise(128, 128, 1.5) * 12)[..., None]
    return a


def rubble():
    a = dirt()
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(40):
        cx, cy = rng.uniform(0, 128), rng.uniform(0, 128)
        r = rng.uniform(3, 9)
        col = (rng.uniform(80, 130), rng.uniform(50, 90), rng.uniform(40, 70)) if rng.random() < 0.5 else (rng.uniform(100, 140),) * 3
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                m = np.hypot(xx - cx + ox, yy - cy + oy) < r
                a[m] = col
    return add_noise(a, 8)


# ---------------------------------------------------------------- metal
def rust_metal():
    a = solid(128, 128, (96, 102, 108))
    add_noise(a, 8)
    stains(a, 8, 8, 34, (128, 72, 36), alpha=0.7, sigma=3)
    stains(a, 5, 4, 14, (70, 40, 24), alpha=0.6, sigma=2)
    for y in (0, 64):
        a[y:y + 2] -= 30
    for x in range(4, 128, 16):
        a[1:3, x:x + 2] += 30
        a[65:67, x:x + 2] += 30
    return a


def panel():
    a = solid(128, 128, (72, 84, 78))
    add_noise(a, 6)
    rect(a, 0, 0, 128, 4, (50, 58, 54))
    rect(a, 0, 124, 128, 128, (50, 58, 54))
    rect(a, 0, 0, 4, 128, (50, 58, 54))
    rect(a, 124, 0, 128, 128, (50, 58, 54))
    for x in range(10, 128, 16):
        for y in (8, 118):
            rect(a, x - 2, y - 2, x + 2, y + 2, (120, 130, 124))
            rect(a, x - 1, y - 1, x + 1, y + 1, (160, 168, 160))
    for y in range(10, 128, 16):
        for x in (8, 118):
            rect(a, x - 2, y - 2, x + 2, y + 2, (120, 130, 124))
    stains(a, 4, 6, 24, (120, 70, 36), alpha=0.5, sigma=3)
    stains(a, 3, 10, 30, (40, 46, 44), alpha=0.4, sigma=4)
    return a


def pipe():
    a = solid(64, 64, (84, 80, 74))
    yy, xx = np.mgrid[0:64, 0:64]
    shade = np.cos((yy / 64.0) * 2 * np.pi)          # boru silindiri gölgesi (dikey eksen boyunca)
    a += (shade * 26)[..., None]
    for x in (0, 32):
        rect(a, x, 0, x + 4, 64, (60, 56, 50))
    stains(a, 4, 4, 14, (122, 70, 34), alpha=0.6, sigma=2)
    return add_noise(a, 5)


def catwalk():
    a = solid(64, 64, (74, 78, 80))
    yy, xx = np.mgrid[0:64, 0:64]
    for ox, oy in ((0, 0), (8, 8)):
        m = ((xx - ox) % 16 < 6) & ((yy - oy) % 16 < 6)
        a[m] += 26
    stains(a, 3, 6, 18, (110, 64, 34), alpha=0.5, sigma=2)
    return add_noise(a, 6)


def corrugated():
    a = solid(128, 128, (88, 84, 78))
    yy, xx = np.mgrid[0:128, 0:128]
    a += (np.cos(xx / 8.0 * np.pi) * 18)[..., None]
    stains(a, 8, 6, 30, (124, 70, 36), alpha=0.7, sigma=3)
    stains(a, 4, 4, 14, (60, 36, 22), alpha=0.6, sigma=2)
    return add_noise(a, 6)


def steel():
    a = solid(64, 64, (44, 50, 56))
    stains(a, 3, 4, 12, (110, 64, 34), alpha=0.5, sigma=2)
    return add_noise(a, 5)


def metal_door():
    a = rust_metal()
    rect(a, 0, 0, 128, 8, (54, 58, 62))
    rect(a, 0, 120, 128, 128, (54, 58, 62))
    for x in range(0, 128, 32):
        rect(a, x, 0, x + 3, 128, (52, 56, 60))
    for y in range(16, 128, 24):
        for x in range(8, 128, 32):
            rect(a, x - 2, y - 2, x + 2, y + 2, (140, 146, 150))
    layer = text_layer(128, 128, "DEPO", "bold", 30, (210, 200, 160), (40, 36, 30), (22, 44), 0)
    paste_rgba(a, layer, 0.6)
    return a


def barrel():
    a = solid(64, 64, (70, 66, 60))
    yy, xx = np.mgrid[0:64, 0:64]
    a += (np.cos(xx / 64.0 * 2 * np.pi) * 18)[..., None]
    for y in (10, 54):
        rect(a, 0, y, 64, y + 3, (50, 46, 42))
    # tehlike bandı + BARUT
    band = (yy >= 22) & (yy < 42)
    hz = ((xx + yy) // 6) % 2 == 0
    a[band & hz] = (200, 150, 30)
    a[band & ~hz] = (30, 28, 26)
    layer = text_layer(64, 64, "BARUT", "bold", 13, (230, 230, 230), (20, 20, 20), (8, 44), 0)
    paste_rgba(a, layer, 0.9)
    stains(a, 5, 4, 14, (120, 68, 32), alpha=0.6, sigma=2)
    return add_noise(a, 5)


def barrel_top():
    a = solid(64, 64, (62, 58, 54))
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.hypot(xx - 32, yy - 32)
    a[r < 30] = (74, 70, 64)
    a[(r > 26) & (r < 30)] = (50, 46, 42)
    a[r < 6] = (54, 50, 46)
    stains(a, 3, 4, 10, (120, 68, 32), alpha=0.6, sigma=2)
    return add_noise(a, 5)


def lamp():
    a = solid(32, 32, (255, 232, 180))
    yy, xx = np.mgrid[0:32, 0:32]
    r = np.hypot(xx - 16, yy - 16)
    a[r > 13] = (90, 86, 80)
    return a


# ---------------------------------------------------------------- ahşap / sandık / çuval
def planks(w=64, h=64, base=(108, 96, 80)):
    a = solid(w, h, base)
    for y in range(0, h, 16):
        a[y:y + 16] += rng.uniform(-12, 12)
        a[y:y + 1] -= 40
        for x in (6, w - 8):
            rect(a, x, y + 7, x + 2, y + 9, (50, 44, 40))
    a += (noise(w, h, 0.7) * 6)[..., None]
    a += (noise(w, h, 4) * 8)[..., None]
    return a


def wood():
    return planks()


def boards():
    a = planks(64, 64, (96, 84, 70))
    stains(a, 3, 6, 18, (50, 44, 40), alpha=0.5, sigma=2)
    return a


def crate():
    a = planks(64, 64, (130, 104, 68))
    rect(a, 0, 0, 64, 6, (92, 70, 44))
    rect(a, 0, 58, 64, 64, (92, 70, 44))
    rect(a, 0, 0, 6, 64, (92, 70, 44))
    rect(a, 58, 0, 64, 64, (92, 70, 44))
    layer = text_layer(64, 64, "TÜTÜN", "mono", 13, (50, 36, 24), (50, 36, 24), (9, 24), 0)
    paste_rgba(a, layer, 0.75)
    return a


def sack():
    a = solid(64, 64, (150, 128, 90))
    yy, xx = np.mgrid[0:64, 0:64]
    a += (np.sin(xx * 0.9) * np.sin(yy * 0.9) * 10)[..., None]
    a += (np.cos(yy / 64.0 * 2 * np.pi) * 16)[..., None]
    rect(a, 0, 0, 64, 3, (90, 76, 52))
    rect(a, 0, 61, 64, 64, (90, 76, 52))
    stains(a, 3, 4, 12, (90, 70, 46), alpha=0.5, sigma=2)
    return add_noise(a, 5)


def tarp():
    a = solid(128, 128, (40, 70, 130))
    add_noise(a, 12)
    yy, xx = np.mgrid[0:128, 0:128]
    a += (np.sin(xx * 0.15 + noise(128, 128, 6, 3)) * 14)[..., None]
    stains(a, 5, 8, 30, (24, 40, 70), alpha=0.5, sigma=4)
    return a


# ---------------------------------------------------------------- pencere / tabela / çöp
def broken_window():
    a = solid(128, 128, (24, 26, 32))
    yy, xx = np.mgrid[0:128, 0:128]
    # camın kalan parçaları (mavimsi parlama)
    for _ in range(9):
        cx, cy = rng.uniform(10, 118), rng.uniform(10, 118)
        ang = rng.uniform(0, 2 * np.pi)
        m = np.ones((128, 128), bool)
        for k in range(3):
            a2 = ang + k * 2.1 + rng.uniform(-0.4, 0.4)
            m &= ((xx - cx) * np.cos(a2) + (yy - cy) * np.sin(a2)) < rng.uniform(8, 22)
        shade = (70 + rng.uniform(-20, 30))
        a[m] = (shade * 0.8, shade * 0.95, shade * 1.15)
    # çerçeve ızgarası
    fr = (58, 52, 46)
    for x in (0, 62, 124):
        rect(a, x, 0, x + 4, 128, fr)
    for y in (0, 42, 84, 124):
        rect(a, 0, y, 128, y + 4, fr)
    rect(a, 0, 0, 128, 6, (74, 68, 60))
    rect(a, 0, 122, 128, 128, (74, 68, 60))
    return add_noise(a, 4)


def sign():
    w, h = 256, 64
    a = solid(w, h, (72, 70, 66))
    add_noise(a, 8)
    rect(a, 0, 0, w, 3, (40, 38, 36))
    rect(a, 0, h - 3, w, h, (40, 38, 36))
    layer = text_layer(w, h, "TÜTÜN FABRİKASI", "serif", 24, (190, 170, 120), (30, 26, 22), None, 0)
    paste_rgba(a, layer, 0.8)
    stains(a, 6, 6, 26, (110, 62, 30), alpha=0.6, sigma=3)
    stains(a, 3, 8, 20, (40, 38, 36), alpha=0.5, sigma=3)
    return a


def site_sign(letter, rgb):
    a = concrete_wall()[:64, :64].copy()
    layer = text_layer(64, 64, letter, "bold", 50, rgb, (20, 20, 20), None, -4)
    paste_rgba(a, layer, 0.85)
    drips(a, layer, rgb, count=6, maxlen=16)
    return a


def garbage():
    a = solid(128, 128, (48, 44, 40))
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(70):
        cx, cy = rng.uniform(0, 128), rng.uniform(0, 128)
        rx, ry = rng.uniform(4, 14), rng.uniform(3, 10)
        pick = rng.random()
        if pick < 0.35:
            col = (rng.uniform(20, 40),) * 3                                  # siyah torba
        elif pick < 0.6:
            col = (rng.uniform(90, 140), rng.uniform(80, 120), rng.uniform(60, 90))   # karton
        elif pick < 0.75:
            col = (rng.uniform(140, 200), rng.uniform(140, 200), rng.uniform(140, 200))  # plastik
        elif pick < 0.88:
            col = (rng.uniform(40, 80), rng.uniform(90, 140), rng.uniform(40, 80))   # şişe
        else:
            col = (rng.uniform(160, 220), rng.uniform(40, 80), rng.uniform(30, 60))   # kırmızı
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                m = (((xx - cx + ox) / rx) ** 2 + ((yy - cy + oy) / ry) ** 2) < 1
                a[m] = col
                edge = m & ((((xx - cx + ox) / rx) ** 2 + ((yy - cy + oy) / ry) ** 2) > 0.6)
                a[edge] = np.array(col) * 0.6
    return add_noise(a, 8)


def fence():
    a = solid(64, 64, (0, 0, 255))        # 255 = şeffaf
    yy, xx = np.mgrid[0:64, 0:64]
    wire = (np.abs(((xx + yy) % 16) - 8) < 1.2) | (np.abs(((xx - yy) % 16) - 8) < 1.2)
    a[wire] = (120, 124, 120)
    a[wire & ((xx + yy) % 2 == 0)] = (96, 100, 96)
    return a


def ladder():
    a = solid(32, 32, (0, 0, 255))
    rect(a, 0, 0, 4, 32, (90, 70, 50))
    rect(a, 28, 0, 32, 32, (90, 70, 50))
    rect(a, 0, 14, 32, 18, (120, 96, 70))
    return a


def trim():
    a = solid(64, 32, (104, 102, 96))
    rect(a, 0, 0, 64, 4, (124, 122, 116))
    rect(a, 0, 28, 64, 32, (70, 68, 64))
    stains(a, 3, 4, 10, (60, 58, 56), alpha=0.5, sigma=2)
    return add_noise(a, 5)


# ---------------------------------------------------------------- sicakcikolata.com
def logo_rgb(w, h):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc_logo.png")
    return np.asarray(Image.open(p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)).astype(float)


def fit_size(text, font, max_w, size):
    while size > 8:
        bb = ImageFont.truetype(FONTS[font], size).getbbox(text)
        if bb[2] - bb[0] + size // 5 <= max_w:
            break
        size -= 1
    return size


def sc_logo():
    """Duvara yapıştırılmış, biraz yıpranmış SÇ logosu afişi (128x240)."""
    a = solid(128, 240, (168, 150, 122))              # kâğıt kenarı
    a[4:236, 4:124] = logo_rgb(120, 232)
    stains(a, 5, 6, 24, (60, 40, 26), alpha=0.3, sigma=3)
    cracks(a, 2, (110, 90, 70), length=40)             # çizik / yırtık
    return add_noise(a, 4)


def sc_text():
    """Eski duvar reklamı: solmuş krem bant üstüne çikolata rengi 'sicakcikolata.com' (512x64)."""
    w, h = 512, 64
    a = solid(w, h, (214, 196, 156))
    add_noise(a, 8)
    rect(a, 0, 0, w, 3, (120, 90, 60)); rect(a, 0, h - 3, w, h, (120, 90, 60))
    size = fit_size("sicakcikolata.com", "bold", w - 32, 44)
    layer = text_layer(w, h, "sicakcikolata.com", "bold", size, (72, 40, 20), (72, 40, 20), None, 0)
    paste_rgba(a, layer, 0.85)
    stains(a, 8, 8, 30, (150, 130, 100), alpha=0.45, sigma=4)     # solma
    stains(a, 3, 6, 16, (90, 60, 40), alpha=0.3, sigma=2)
    return a


TEXTURES = {
    "tf_brick": brick, "tf_brick2": brick_dark, "tf_plaster": plaster,
    "tf_graf1": graf1, "tf_graf2": graf2, "tf_graf3": graf3, "tf_graf4": graf4,
    "tf_cwall": concrete_wall, "tf_concrete": concrete_floor, "tf_asphalt": asphalt,
    "tf_dirt": dirt, "tf_gravel": gravel, "tf_rubble": rubble,
    "tf_metal": rust_metal, "tf_panel": panel, "tf_pipe": pipe, "tf_catwalk": catwalk,
    "tf_roof": corrugated, "tf_steel": steel, "tf_door": metal_door,
    "tf_barrel": barrel, "tf_barreltop": barrel_top, "tf_lamp": lamp,
    "tf_wood": wood, "tf_boards": boards, "tf_crate": crate, "tf_sack": sack, "tf_tarp": tarp,
    "tf_window": broken_window, "tf_sign": sign,
    "tf_signa": lambda: site_sign("A", (230, 60, 40)), "tf_signb": lambda: site_sign("B", (240, 200, 40)),
    "tf_garbage": garbage, "{tf_fence": fence, "{tf_ladder": ladder, "tf_trim": trim,
    "tf_sclogo": sc_logo, "tf_scyazi": sc_text,                  # sicakcikolata.com afişi ve duvar reklamı
}


# ---------------------------------------------------------------- WAD3
def to_miptex(name, rgb):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    assert w % 16 == 0 and h % 16 == 0, name
    img = Image.fromarray(rgb, "RGB")
    if name.startswith("{"):
        q = img.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = q.getpalette()[:255 * 3] + [0, 0, 255]
        arr = np.array(q)
        blue = np.all(rgb == (0, 0, 255), axis=2)
        arr[blue] = 255
        mips = [arr.astype(np.uint8)]
        for k in (2, 4, 8):
            mips.append(arr[::k, ::k].astype(np.uint8))
    else:
        q = img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = q.getpalette()[:768]
        pal += [0] * (768 - len(pal))
        mips = [np.array(q, dtype=np.uint8)]
        for k in (2, 4, 8):
            small = img.resize((w // k, h // k), Image.Resampling.BOX)
            mips.append(np.array(small.quantize(palette=q, dither=Image.Dither.NONE), dtype=np.uint8))
    hdr = 16 + 4 * 6
    offs, data, pos = [], b"", hdr
    for m in mips:
        offs.append(pos)
        b = m.tobytes()
        data += b
        pos += len(b)
    nm = name.encode()[:15].ljust(16, b"\0")
    blob = nm + struct.pack("<II", w, h) + struct.pack("<4I", *offs) + data
    blob += struct.pack("<H", 256) + bytes(pal) + b"\0\0"
    return blob


def write_wad(path, previews_dir=None):
    lumps, sizes = [], {}
    for name, fn in TEXTURES.items():
        rgb = fn()
        sizes[name] = (rgb.shape[1], rgb.shape[0])
        lumps.append((name, to_miptex(name, rgb)))
        if previews_dir:
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
                f"{previews_dir}/{name.strip('!{')}.png")
    out = bytearray(b"WAD3" + struct.pack("<ii", len(lumps), 0))
    entries = []
    for name, blob in lumps:
        entries.append((len(out), len(blob), name))
        out += blob
        while len(out) % 4:
            out += b"\0"
    dirofs = len(out)
    for pos, size, name in entries:
        out += struct.pack("<iiibbh", pos, size, size, 0x43, 0, 0) + name.encode()[:15].ljust(16, b"\0")
    struct.pack_into("<i", out, 8, dirofs)
    open(path, "wb").write(out)
    return sizes


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    prev = sys.argv[1] if len(sys.argv) > 1 else None
    if prev:
        os.makedirs(prev, exist_ok=True)
    sizes = write_wad(os.path.join(here, "tutun.wad"), prev)
    print(len(sizes), "doku ->", os.path.join(here, "tutun.wad"))
