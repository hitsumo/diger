"""de_hamamonu için yordamsal dokular -> hamamonu.wad (WAD3).

Restore edilmiş Osmanlı mahallesi: renkli sıvalı cumbalı evler, ahşap pencere ve kapılar,
kiremit çatı, arnavut kaldırımı, kesme taş saat kulesi, kurşun kubbeli hamam, dükkân
tabelaları, tenteler, kilimler, ağaç yaprağı, Hacettepe cephesi.

Kullanım: python textures.py [onizleme_dizini]
"""
import os
import struct
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(1071)
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
def plaster(rgb, dirt=0.35):
    a = solid(128, 128, rgb)
    add_noise(a, 6)
    stains(a, 4, 10, 36, tuple(c * 0.72 for c in rgb), alpha=dirt, sigma=4)
    stains(a, 2, 6, 14, (120, 112, 100), alpha=0.25, sigma=2)
    cracks(a, 2, tuple(c * 0.6 for c in rgb), length=40)
    return a


def plaster_white():
    return plaster((226, 220, 206))


def plaster_ochre():
    return plaster((214, 176, 110))


def plaster_pink():
    return plaster((214, 168, 150))


def plaster_blue():
    return plaster((170, 192, 206))


def plaster_green():
    return plaster((168, 190, 150))


def inner_wall():
    a = plaster((232, 226, 212), 0.2)
    rect(a, 0, 100, 128, 108, (120, 92, 60))       # süpürgelik
    return a


def rubble_stone():
    a = solid(128, 128, (142, 134, 118))
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(40):
        cx, cy = rng.uniform(0, 128), rng.uniform(0, 128)
        rx, ry = rng.uniform(10, 24), rng.uniform(7, 14)
        col = np.array((142, 134, 118)) + rng.uniform(-22, 22)
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                m = (((xx - cx + ox) / rx) ** 2 + ((yy - cy + oy) / ry) ** 2) < 1
                a[m] = col
                e = m & ((((xx - cx + ox) / rx) ** 2 + ((yy - cy + oy) / ry) ** 2) > 0.7)
                a[e] = col * 0.75
    return add_noise(a, 7)


def ashlar():
    return blocks(128, 128, 64, 32, (196, 186, 164), (150, 142, 126), var=8, mortar_w=2, bevel=2)


def ashlar_dark():
    return blocks(128, 128, 64, 32, (166, 156, 136), (120, 112, 100), var=8, mortar_w=2, bevel=2)


def garden_wall():
    a = rubble_stone()
    rect(a, 0, 0, 128, 10, (178, 170, 154))
    rect(a, 0, 10, 128, 12, (110, 104, 92))
    return a


def wood(w=64, h=64, base=(128, 92, 56)):
    a = solid(w, h, base)
    for y in range(0, h, 16):
        a[y:y + 16] += rng.uniform(-10, 10)
        a[y:y + 1] -= 34
    a += (noise(w, h, 0.7) * 5)[..., None]
    a += (noise(w, h, 4) * 8)[..., None]
    return a


def wood_dark():
    return wood(64, 64, (96, 66, 40))


def window():
    """Ahşap kepenkli pencere, 64x128."""
    a = solid(64, 128, (226, 220, 206))
    add_noise(a, 5)
    rect(a, 6, 6, 58, 122, (96, 66, 40))                 # çerçeve
    rect(a, 10, 10, 54, 118, (52, 58, 70))               # cam
    yy, xx = np.mgrid[0:128, 0:64]
    glare = ((xx + yy * 0.5) % 40 < 10) & (xx >= 10) & (xx < 54) & (yy >= 10) & (yy < 118)
    a[glare] = (78, 90, 108)
    rect(a, 30, 10, 34, 118, (96, 66, 40))               # orta kayıt
    for y in (44, 80):
        rect(a, 10, y, 54, y + 4, (96, 66, 40))
    for x in (0, 60):                                     # kepenk kenarları
        rect(a, x, 4, x + 4, 124, (80, 54, 34))
    rect(a, 2, 120, 62, 126, (120, 112, 100))             # denizlik
    return a


def lattice_window():
    """Cumba penceresi: kafes (kafes ahşap), 64x64."""
    a = solid(64, 64, (60, 64, 76))
    rect(a, 0, 0, 64, 4, (96, 66, 40)); rect(a, 0, 60, 64, 64, (96, 66, 40))
    rect(a, 0, 0, 4, 64, (96, 66, 40)); rect(a, 60, 0, 64, 64, (96, 66, 40))
    yy, xx = np.mgrid[0:64, 0:64]
    lat = (xx % 8 < 2) | (yy % 8 < 2)
    a[lat & (xx >= 4) & (xx < 60) & (yy >= 4) & (yy < 60)] = (124, 90, 54)
    return a


def door():
    a = solid(64, 128, (96, 66, 40))
    rect(a, 31, 0, 33, 128, (60, 40, 24))
    for x0 in (6, 38):
        for y0 in (10, 52, 94):
            rect(a, x0, y0, x0 + 20, y0 + 30, (116, 82, 50))
            rect(a, x0 + 2, y0 + 2, x0 + 18, y0 + 28, (104, 72, 44))
    rect(a, 22, 62, 26, 70, (190, 160, 70)); rect(a, 38, 62, 42, 70, (190, 160, 70))   # tokmak
    yy, xx = np.mgrid[0:128, 0:64]
    arch = ((xx - 32) ** 2 + (yy - 24) ** 2 > 30 * 30) & (yy < 24)
    a[arch] = (196, 186, 164)
    return add_noise(a, 4)


def shopfront():
    """Dükkân vitrini: raflarda bakır ve lokum, 128x128."""
    a = solid(128, 128, (96, 66, 40))
    rect(a, 8, 8, 120, 120, (54, 58, 66))
    for y0 in (24, 64, 100):
        rect(a, 10, y0 + 18, 118, y0 + 22, (120, 86, 50))
        for i in range(5):
            x0 = 14 + i * 21
            pick = (i + y0) % 3
            col = (190, 120, 60) if pick == 0 else ((210, 70, 90) if pick == 1 else (200, 180, 90))
            rect(a, x0, y0, x0 + 14, y0 + 18, col)
            rect(a, x0 + 3, y0 + 3, x0 + 11, y0 + 15, tuple(c * 1.15 for c in col))
    yy, xx = np.mgrid[0:128, 0:128]
    glare = ((xx - yy) % 56 < 8) & (xx >= 8) & (xx < 120) & (yy >= 8) & (yy < 120)
    a[glare] = a[glare] * 0.6 + np.array((150, 160, 180)) * 0.4
    rect(a, 60, 8, 68, 120, (96, 66, 40))
    return a


def cumba_bottom():
    a = wood(64, 64, (110, 78, 48))
    for x in (0, 20, 40, 60):
        rect(a, x, 0, x + 3, 64, (70, 48, 30))
    return a


def eave():
    a = wood(64, 32, (110, 78, 48))
    for x in range(0, 64, 16):
        rect(a, x, 20, x + 6, 32, (70, 48, 30))
    return a


def roof_tile():
    a = solid(128, 128, (176, 86, 58))
    yy, xx = np.mgrid[0:128, 0:128]
    a += (np.cos(xx / 16.0 * 2 * np.pi) * 16)[..., None]
    for y in range(0, 128, 16):
        a[y:y + 2] -= 36
        a[y + 2:y + 4] += 14
    stains(a, 5, 6, 24, (110, 100, 80), alpha=0.35, sigma=3)
    stains(a, 4, 4, 14, (90, 110, 70), alpha=0.3, sigma=2)
    return add_noise(a, 8)


def lead_dome():
    a = solid(128, 128, (118, 122, 128))
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 32 < 2)] -= 30
    for y in range(0, 128, 24):
        a[y:y + 1] -= 24
    stains(a, 4, 8, 30, (90, 96, 104), alpha=0.4, sigma=4)
    return add_noise(a, 5)


# ---------------------------------------------------------------- zeminler
def cobble():
    a = solid(128, 128, (118, 112, 102))
    yy, xx = np.mgrid[0:128, 0:128]
    for r in range(8):
        off = 8 if r % 2 else 0
        for c in range(-1, 9):
            cx, cy = c * 16 + off + 8, r * 16 + 8
            d = np.maximum(np.abs(((xx - cx + 64) % 128) - 64), np.abs(((yy - cy + 64) % 128) - 64))
            m = d < 7
            a[m] = np.array((118, 112, 102)) + rng.uniform(-18, 18)
            a[(d >= 5) & (d < 7)] -= 18
    a[(xx % 16 < 2) | (yy % 16 < 2)] = (82, 78, 72)
    return add_noise(a, 8)


def paving():
    a = solid(128, 128, (186, 178, 160))
    for y in (0, 64):
        a[y:y + 2] -= 40
    for x in (0, 64):
        a[:, x:x + 2] -= 40
    stains(a, 5, 8, 30, (140, 132, 118), alpha=0.35, sigma=4)
    return add_noise(a, 7)


def paving_red():
    a = solid(128, 128, (164, 112, 92))
    yy, xx = np.mgrid[0:128, 0:128]
    a[((xx // 32 + yy // 32) % 2 == 0)] = (186, 178, 160)
    a[(xx % 32 < 2) | (yy % 32 < 2)] -= 40
    return add_noise(a, 7)


def wood_floor():
    a = wood(128, 128, (150, 112, 68))
    for x in range(0, 128, 64):
        rect(a, x, 0, x + 2, 128, (90, 64, 40))
    return a


def kilim():
    a = solid(128, 128, (150, 36, 40))
    yy, xx = np.mgrid[0:128, 0:128]
    d = np.abs(xx - 64) + np.abs(yy - 64)
    for r0, r1, col in ((56, 64, (40, 30, 70)), (44, 50, (220, 190, 90)), (30, 38, (40, 30, 70)), (10, 20, (220, 190, 90))):
        a[(d >= r0) & (d < r1)] = col
    rect(a, 0, 0, 128, 8, (220, 190, 90)); rect(a, 0, 120, 128, 128, (220, 190, 90))
    a[((xx + yy) % 16 < 2) & ((d < 10) | (d >= 64))] = (40, 30, 70)
    return add_noise(a, 6)


def ceiling_wood():
    a = wood(128, 128, (138, 104, 64))
    for x in range(0, 128, 32):
        rect(a, x, 0, x + 6, 128, (96, 66, 40))
    return a


def marble():
    a = solid(128, 128, (214, 212, 206))
    a += (noise(128, 128, 3) * 16)[..., None]
    cracks(a, 4, (150, 148, 142), length=60)
    for y in (0, 64):
        a[y:y + 2] -= 30
    for x in (0, 64):
        a[:, x:x + 2] -= 30
    return a


def grass():
    a = solid(128, 128, (92, 122, 60))
    a += (noise(128, 128, 0.6) * 20)[..., None]
    a += (noise(128, 128, 3) * 12)[..., None]
    a[..., 1] += noise(128, 128, 1.5) * 10
    return a


def dirt():
    a = solid(128, 128, (132, 112, 84))
    add_noise(a, 12)
    stains(a, 4, 10, 30, (100, 84, 62), alpha=0.4, sigma=4)
    return a


def hedge():
    a = solid(64, 64, (54, 92, 44))
    a += (noise(64, 64, 0.6) * 26)[..., None]
    a[..., 1] += noise(64, 64, 1.2) * 14
    return a


def bark():
    a = solid(64, 64, (96, 76, 56))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(xx % 8 < 2)] -= 26
    a += (noise(64, 64, 2) * 12)[..., None]
    return a


def leaves():
    a = solid(128, 128, (0, 0, 255))
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(140):
        cx, cy, r = rng.uniform(0, 128), rng.uniform(0, 128), rng.uniform(4, 9)
        g = rng.uniform(80, 140)
        col = (g * 0.55, g, g * 0.4)
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                m = np.hypot(xx - cx + ox, yy - cy + oy) < r
                a[m] = col
    return a


def water():
    a = solid(64, 64, (60, 110, 130))
    a += (noise(64, 64, 2) * 18)[..., None]
    a += (noise(64, 64, 0.8) * 8)[..., None]
    return a


# ---------------------------------------------------------------- tabelalar, özel
def sign(text, w, h, bg, fg, font="serif", size=None, border=(60, 40, 24)):
    a = solid(w, h, bg)
    add_noise(a, 4)
    rect(a, 0, 0, w, 3, border); rect(a, 0, h - 3, w, h, border)
    rect(a, 0, 0, 3, h, border); rect(a, w - 3, 0, w, h, border)
    size = size or int(h * 0.55)
    while size > 8:                                   # genişliğe sığdır
        f = ImageFont.truetype(FONTS[font], size)
        bb = f.getbbox(text)
        if bb[2] - bb[0] <= w - 20:
            break
        size -= 1
    layer = text_layer(w, h, text, font, size, fg, border, None, 0)
    paste_rgba(a, layer, 0.95)
    return a


def clock():
    a = solid(128, 128, (236, 232, 220))
    yy, xx = np.mgrid[0:128, 0:128]
    r = np.hypot(xx - 64, yy - 64)
    a[r > 60] = (166, 156, 136)
    a[(r > 56) & (r <= 60)] = (40, 36, 32)
    for k in range(12):
        ang = k * np.pi / 6
        px, py = 64 + 48 * np.sin(ang), 64 - 48 * np.cos(ang)
        a[(np.hypot(xx - px, yy - py) < 3)] = (30, 30, 30)
    for ang, L, w in ((np.radians(300), 30, 3), (np.radians(60), 44, 2)):      # akrep 10, yelkovan 2
        t = np.linspace(0, 1, 60)
        for tt in t:
            px, py = 64 + L * tt * np.sin(ang), 64 - L * tt * np.cos(ang)
            a[(np.hypot(xx - px, yy - py) < w)] = (30, 30, 30)
    a[r < 4] = (30, 30, 30)
    return a


def hacettepe_wall():
    """Modern cephe: taş kaplama + pencere bandı, 128x256 (zemin + 1 kat)."""
    a = blocks(128, 256, 64, 32, (200, 190, 170), (160, 152, 136), var=5, mortar_w=1)
    for y0 in (40, 168):
        rect(a, 12, y0, 116, y0 + 72, (48, 60, 80))
        yy, xx = np.mgrid[0:256, 0:128]
        g = ((xx + yy * 0.4) % 44 < 10) & (yy >= y0) & (yy < y0 + 72) & (xx >= 12) & (xx < 116)
        a[g] = (90, 110, 140)
        rect(a, 62, y0, 66, y0 + 72, (160, 160, 160))
        rect(a, 12, y0 + 34, 116, y0 + 38, (160, 160, 160))
    return a


def glass():
    a = solid(64, 64, (48, 60, 80))
    yy, xx = np.mgrid[0:64, 0:64]
    a[((xx + yy * 0.4) % 32 < 8)] = (90, 110, 140)
    return a


def concrete():
    a = solid(128, 128, (150, 148, 142))
    add_noise(a, 7)
    stains(a, 4, 8, 30, (110, 108, 102), alpha=0.35, sigma=4)
    return a


def awning():
    a = solid(64, 64, (210, 40, 40))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(xx // 8) % 2 == 0] = (236, 230, 220)
    return add_noise(a, 6)


def awning_green():
    a = solid(64, 64, (40, 110, 60))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(xx // 8) % 2 == 0] = (236, 230, 220)
    return add_noise(a, 6)


def metal():
    a = solid(64, 64, (70, 74, 78))
    return add_noise(a, 5)


def lamp():
    a = solid(32, 32, (255, 240, 200))
    yy, xx = np.mgrid[0:32, 0:32]
    a[np.hypot(xx - 16, yy - 16) > 13] = (60, 56, 50)
    return a


def bench():
    a = wood(64, 32, (118, 84, 48))
    for x in range(0, 64, 8):
        rect(a, x, 0, x + 1, 32, (70, 48, 30))
    return a


def poster():
    a = solid(64, 128, (240, 232, 210))
    rect(a, 4, 4, 60, 60, (190, 60, 50))
    rect(a, 4, 60, 60, 124, (60, 90, 150))
    layer = text_layer(64, 128, "HAMAM\nÖNÜ", "bold", 14, (250, 250, 250), (30, 30, 30), (6, 16), 0)
    paste_rgba(a, layer, 0.95)
    layer = text_layer(64, 128, "FESTİVAL", "bold", 10, (250, 240, 120), (30, 30, 30), (4, 96), 0)
    paste_rgba(a, layer, 0.95)
    stains(a, 2, 6, 14, (120, 110, 90), alpha=0.3, sigma=2)
    return a


def site_sign(letter, rgb):
    a = plaster((226, 220, 206), 0.2)[:64, :64].copy()
    layer = text_layer(64, 64, letter, "bold", 50, rgb, (20, 20, 20), None, 0)
    paste_rgba(a, layer, 0.9)
    return a


def fence():
    a = solid(64, 64, (0, 0, 255))
    rect(a, 0, 4, 64, 8, (60, 56, 50)); rect(a, 0, 40, 64, 44, (60, 56, 50))
    for x in range(2, 64, 8):
        rect(a, x, 0, x + 2, 64, (60, 56, 50))
    return a


def ladder():
    a = solid(32, 32, (0, 0, 255))
    rect(a, 0, 0, 4, 32, (96, 66, 40)); rect(a, 28, 0, 32, 32, (96, 66, 40))
    rect(a, 0, 14, 32, 18, (124, 90, 54))
    return a


def bus_sign():
    a = solid(64, 64, (40, 90, 160))
    layer = text_layer(64, 64, "EGO", "bold", 22, (250, 250, 250), (20, 40, 80), None, 0)
    paste_rgba(a, layer, 0.95)
    rect(a, 0, 0, 64, 3, (230, 230, 230)); rect(a, 0, 61, 64, 64, (230, 230, 230))
    return a


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
    """Asılı dükkân tabelası: SÇ logosu, ince altın çerçeve (128x240)."""
    a = solid(128, 240, (204, 164, 72))
    a[4:236, 4:124] = logo_rgb(120, 232)
    return a


def sc_pano():
    """Reklam panosu 256x128: solda logo, sağda 'sicakcikolata.com', altta çikolata damlaları."""
    w, h = 256, 128
    yy, xx = np.mgrid[0:h, 0:w]
    t = (yy / h)[..., None]
    a = np.array((150, 82, 20), float) * (1 - t) + np.array((84, 42, 8), float) * t
    a = a + np.zeros((h, w, 3))
    a[8:120, 10:78] = logo_rgb(68, 112)
    size = fit_size("sicakcikolata.com", "bold", 156, 26)
    layer = text_layer(w, h, "sicakcikolata.com", "bold", size, (250, 232, 190), (40, 20, 8), None, 0)
    # yazıyı sağ bölüme kaydır: katmanı 86 px sağa ötele (merkez 128 -> 164)
    layer = np.roll(layer, 36, axis=1)
    paste_rgba(a, layer, 1.0)
    for cx, wd, ln in ((100, 7, 22), (124, 5, 14), (150, 8, 30), (180, 6, 18), (206, 7, 26), (232, 5, 12)):   # damlalar
        a[h - ln:h, cx - wd:cx + wd] = (40, 20, 8)
        r = np.hypot(xx - cx, yy - (h - ln)) < wd
        a[r] = (40, 20, 8)
    rect(a, 0, 0, w, 3, (40, 20, 8)); rect(a, 0, h - 3, w, h, (40, 20, 8))
    rect(a, 0, 0, 3, h, (40, 20, 8)); rect(a, w - 3, 0, w, h, (40, 20, 8))
    return a


def awning_sc():
    a = solid(64, 64, (120, 64, 24))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(xx // 8) % 2 == 0] = (240, 224, 196)
    return add_noise(a, 6)


TEXTURES = {
    "ho_pl_w": plaster_white, "ho_pl_y": plaster_ochre, "ho_pl_p": plaster_pink, "ho_pl_b": plaster_blue,
    "ho_pl_g": plaster_green, "ho_inner": inner_wall, "ho_stone": rubble_stone, "ho_ashlar": ashlar,
    "ho_ashlar2": ashlar_dark, "ho_gwall": garden_wall, "ho_wood": wood, "ho_wood2": wood_dark,
    "ho_window": window, "ho_lattice": lattice_window, "ho_door": door, "ho_shop": shopfront,
    "ho_cumba": cumba_bottom, "ho_eave": eave, "ho_tile": roof_tile, "ho_dome": lead_dome,
    "ho_cobble": cobble, "ho_paving": paving, "ho_paving2": paving_red, "ho_floor": wood_floor,
    "ho_kilim": kilim, "ho_ceil": ceiling_wood, "ho_marble": marble, "ho_grass": grass, "ho_dirt": dirt,
    "ho_hedge": hedge, "ho_bark": bark, "{ho_leaves": leaves, "!ho_water": water,
    "ho_clock": clock, "ho_hacet": hacettepe_wall, "ho_glass": glass, "ho_concrete": concrete,
    "ho_awning": awning, "ho_awning2": awning_green, "ho_metal": metal, "ho_lamp": lamp, "ho_bench": bench,
    "ho_poster": poster, "ho_signa": lambda: site_sign("A", (230, 60, 40)), "ho_signb": lambda: site_sign("B", (240, 200, 40)),
    "{ho_fence": fence, "{ho_ladder": ladder, "ho_bus": bus_sign,
    "ho_s_cay": lambda: sign("ÇAY BAHÇESİ", 256, 64, (120, 36, 36), (240, 220, 160)),
    "ho_s_bakir": lambda: sign("BAKIRCI", 128, 32, (60, 60, 70), (230, 170, 80), "bold"),
    "ho_s_lokum": lambda: sign("LOKUM & KAHVE", 256, 64, (40, 90, 70), (240, 230, 200)),
    "ho_s_hamam": lambda: sign("KARACABEY HAMAMI", 256, 64, (166, 156, 136), (40, 36, 32)),
    "ho_s_hacet": lambda: sign("HACETTEPE ÜNİVERSİTESİ", 512, 64, (200, 190, 170), (40, 50, 80), "bold", 30, (150, 140, 120)),
    "ho_s_kule": lambda: sign("SAAT KULESİ MEYDANI", 256, 32, (60, 60, 70), (240, 240, 240), "bold"),
    "ho_s_hacetyol": lambda: sign("HACETTEPE ÖNÜ", 256, 32, (60, 60, 70), (240, 240, 240), "bold"),
    # sicakcikolata.com: asılı logo tabelası, reklam panosu, dükkân yazısı, kafe tentesi
    "ho_sclogo": sc_logo, "ho_scpano": sc_pano,
    "ho_s_sc": lambda: sign("sicakcikolata.com", 256, 32, (74, 40, 18), (250, 232, 190), "bold", None, (36, 18, 6)),
    "ho_awning_sc": awning_sc,
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
    sizes = write_wad(os.path.join(here, "hamamonu.wad"), prev)
    print(len(sizes), "doku ->", os.path.join(here, "hamamonu.wad"))
