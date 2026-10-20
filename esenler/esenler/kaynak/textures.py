"""Esenler Otogarı (cs_esenler + de_esenler) için yordamsal dokular -> esenler.wad (WAD3).

90'lar otogarı: terrazzo ve kahverengi fayans zeminler, isli beton, yazıhane cepheleri,
kepenkler, otobüs ve metro vagonu kaplamaları, tuvalet fayansı, raf/koli/bavul, tabelalar.

Kullanım: python textures.py [onizleme_dizini]
"""
import os
import struct
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(1994)
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


# ---------------------------------------------------------------- zeminler
def asphalt():
    a = solid(128, 128, (64, 62, 60))
    add_noise(a, 8)
    a += (noise(128, 128, 0.5) * 7)[..., None]
    stains(a, 5, 10, 40, (40, 38, 38), alpha=0.5, sigma=5)        # yağ lekeleri
    stains(a, 3, 8, 24, (84, 80, 74), alpha=0.3, sigma=4)
    cracks(a, 3, (44, 42, 40), length=60)
    return a


def white():
    a = solid(32, 32, (214, 210, 200))
    return add_noise(a, 6)


def yellow():
    a = solid(32, 32, (214, 178, 40))
    return add_noise(a, 6)


def curb():
    a = solid(64, 32, (150, 148, 140))
    rect(a, 0, 0, 64, 4, (176, 174, 166))
    rect(a, 0, 28, 64, 32, (110, 108, 102))
    stains(a, 3, 4, 10, (90, 88, 84), alpha=0.4, sigma=2)
    return add_noise(a, 6)


def terrazzo():
    """Bej terrazzo (90'lar gar salonu), 64'lük karo derzleri, aşınma izleri."""
    a = solid(128, 128, (196, 184, 160))
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(260):
        cx, cy, r = rng.uniform(0, 128), rng.uniform(0, 128), rng.uniform(1, 3.5)
        col = (rng.uniform(110, 150), rng.uniform(100, 130), rng.uniform(90, 120)) if rng.random() < 0.6 else (rng.uniform(220, 240),) * 3
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                a[np.hypot(xx - cx + ox, yy - cy + oy) < r] = col
    a[(xx % 64 < 2) | (yy % 64 < 2)] = (130, 122, 108)
    stains(a, 5, 10, 36, (140, 128, 108), alpha=0.35, sigma=5)
    return add_noise(a, 4)


def tile_brown():
    """Kahverengi seramik, 32'lik karo (alt kat koridorları)."""
    a = solid(128, 128, (118, 76, 50))
    yy, xx = np.mgrid[0:128, 0:128]
    for r in range(4):
        for c in range(4):
            a[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] += rng.uniform(-12, 12)
    a[(xx % 32 < 2) | (yy % 32 < 2)] = (176, 166, 150)
    a[(xx % 32 >= 2) & (xx % 32 < 4)] += 10
    stains(a, 5, 8, 30, (70, 46, 30), alpha=0.4, sigma=4)
    return add_noise(a, 5)


def tile_wet():
    """Tuvalet zemini: zeytin-gri karo, ıslak koyu lekeler."""
    a = solid(128, 128, (112, 116, 100))
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 32 < 2) | (yy % 32 < 2)] = (70, 72, 64)
    stains(a, 6, 12, 40, (56, 60, 52), alpha=0.6, sigma=5)         # ıslak
    stains(a, 4, 6, 18, (140, 130, 90), alpha=0.35, sigma=3)       # sarımsı
    return add_noise(a, 5)


def granite():
    a = solid(128, 128, (132, 130, 128))
    a += (noise(128, 128, 0.5) * 14)[..., None]
    a += (noise(128, 128, 2) * 8)[..., None]
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 64 < 1) | (yy % 64 < 1)] = (90, 88, 86)
    return a


def concrete():
    a = solid(128, 128, (138, 136, 130))
    add_noise(a, 8)
    stains(a, 5, 10, 36, (98, 96, 92), alpha=0.4, sigma=4)
    cracks(a, 2, (90, 88, 84), length=50)
    return a


def rubble():
    a = concrete()
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(40):
        cx, cy = rng.uniform(0, 128), rng.uniform(0, 128)
        r = rng.uniform(3, 10)
        col = (rng.uniform(100, 150),) * 3 if rng.random() < 0.7 else (rng.uniform(120, 160), rng.uniform(80, 100), rng.uniform(60, 80))
        for ox in (-128, 0, 128):
            for oy in (-128, 0, 128):
                a[np.hypot(xx - cx + ox, yy - cy + oy) < r] = col
    return add_noise(a, 8)


# ---------------------------------------------------------------- duvarlar
def wall():
    """Bej yağlı boya duvar, altta kir bandı ve sürtünme izleri."""
    a = solid(128, 128, (206, 192, 160))
    add_noise(a, 5)
    rect(a, 0, 96, 128, 100, (120, 96, 70))                        # bordür
    a[100:128] = a[100:128] * 0.85
    stains(a, 5, 8, 28, (150, 134, 104), alpha=0.35, sigma=4)
    stains(a, 3, 4, 12, (90, 80, 64), alpha=0.3, sigma=2)
    cracks(a, 1, (150, 136, 110), length=40)
    return a


def wall_up():
    a = solid(128, 128, (212, 198, 168))
    add_noise(a, 5)
    stains(a, 4, 10, 36, (170, 156, 128), alpha=0.3, sigma=5)
    return a


def wtile():
    """Beyaz tuvalet fayansı 16'lık, derz kirli, sarımsı lekeler."""
    a = solid(128, 128, (222, 222, 214))
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 16 < 1) | (yy % 16 < 1)] = (150, 146, 130)
    for r in range(8):
        for c in range(8):
            a[r * 16 + 1:(r + 1) * 16, c * 16 + 1:(c + 1) * 16] += rng.uniform(-6, 6)
    stains(a, 6, 6, 26, (170, 160, 110), alpha=0.35, sigma=3)
    stains(a, 3, 4, 14, (120, 110, 90), alpha=0.3, sigma=2)
    return add_noise(a, 3)


def mtile():
    """Metro istasyonu duvarı: beyaz fayans, ortada kırmızı bant."""
    a = solid(128, 128, (228, 226, 220))
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 32 < 1) | (yy % 32 < 1)] = (170, 168, 160)
    rect(a, 0, 56, 128, 72, (190, 30, 40))
    stains(a, 3, 8, 24, (180, 176, 166), alpha=0.3, sigma=4)
    return add_noise(a, 3)


def facade():
    """Çevre bina üst kat: beton panel + pencere bandı (128x256), isli."""
    a = solid(128, 256, (168, 160, 146))
    add_noise(a, 6)
    for y0 in (24, 152):
        rect(a, 8, y0, 120, y0 + 72, (44, 52, 64))
        yy, xx = np.mgrid[0:256, 0:128]
        g = ((xx + yy * 0.35) % 40 < 8) & (yy >= y0) & (yy < y0 + 72) & (xx >= 8) & (xx < 120)
        a[g] = (80, 96, 118)
        rect(a, 62, y0, 66, y0 + 72, (140, 140, 140))
        rect(a, 8, y0 + 34, 120, y0 + 38, (140, 140, 140))
        rect(a, 4, y0 + 72, 124, y0 + 78, (120, 114, 104))
    stains(a, 6, 8, 30, (110, 104, 94), alpha=0.4, sigma=4)       # is
    for y in (0, 128):
        a[y:y + 2] -= 30
    return a


def fac_low():
    a = solid(128, 128, (132, 126, 116))
    add_noise(a, 6)
    rect(a, 0, 0, 128, 8, (100, 96, 90))
    stains(a, 6, 8, 30, (84, 80, 74), alpha=0.5, sigma=4)
    cracks(a, 2, (90, 86, 80), length=40)
    return a


def pillar():
    a = solid(64, 128, (150, 146, 138))
    add_noise(a, 6)
    rect(a, 0, 0, 2, 128, (110, 106, 100)); rect(a, 62, 0, 64, 128, (110, 106, 100))
    stains(a, 4, 6, 20, (96, 92, 86), alpha=0.45, sigma=3)
    a[96:128] = a[96:128] * 0.8
    return a


def shutter(graffiti=False):
    a = solid(128, 128, (138, 140, 142))
    yy, xx = np.mgrid[0:128, 0:128]
    a += (np.cos(yy / 8.0 * 2 * np.pi) * 16)[..., None]
    a[(yy % 8 < 1)] -= 30
    stains(a, 7, 6, 28, (124, 70, 36), alpha=0.6, sigma=3)         # pas
    stains(a, 4, 8, 30, (90, 92, 94), alpha=0.4, sigma=4)
    if graffiti:
        layer = text_layer(128, 128, "EsEs", "oblique", 44, (230, 60, 60), (30, 10, 10), (8, 30), -6)
        paste_rgba(a, layer, 0.85)
        drips(a, layer, (230, 60, 60), 8, 20)
    return add_noise(a, 5)


def shop():
    """Kapalı dükkân camekânı: tozlu cam, içeride karanlık raflar, KAPALI kâğıdı."""
    a = solid(128, 128, (54, 56, 60))
    for y0 in (30, 70, 104):
        rect(a, 8, y0, 120, y0 + 3, (80, 70, 56))
        for i in range(5):
            if rng.random() < 0.5:
                x0 = 12 + i * 22
                rect(a, x0, y0 - 14, x0 + 12, y0, (rng.uniform(60, 110), rng.uniform(50, 90), rng.uniform(40, 70)))
    yy, xx = np.mgrid[0:128, 0:128]
    glare = ((xx + yy * 0.5) % 48 < 10)
    a[glare] = a[glare] * 0.7 + np.array((150, 156, 164)) * 0.3
    a = a * 0.85 + np.array((150, 140, 120)) * 0.15                  # toz
    rect(a, 0, 0, 128, 6, (90, 86, 80)); rect(a, 0, 122, 128, 128, (90, 86, 80))
    rect(a, 0, 0, 6, 128, (90, 86, 80)); rect(a, 122, 0, 128, 128, (90, 86, 80))
    rect(a, 60, 0, 68, 128, (90, 86, 80))
    layer = text_layer(128, 128, "KAPALI", "bold", 16, (40, 30, 24), (230, 226, 210), (18, 40), -8)
    rect(a, 12, 34, 72, 64, (226, 220, 204))
    paste_rgba(a, layer, 1.0)
    return add_noise(a, 4)


def office():
    """Yazıhane cephesi 256x128: cam kapı + vitrin, içeride gişe ve şehir listesi."""
    w, h = 256, 128
    a = solid(w, h, (96, 90, 82))
    rect(a, 6, 6, 250, 122, (60, 70, 84))
    # iç: gişe
    rect(a, 20, 70, 110, 122, (120, 84, 48)); rect(a, 20, 66, 110, 72, (160, 150, 130))
    # şehir listesi (sağ vitrin)
    rect(a, 140, 18, 240, 100, (236, 232, 220))
    f = ImageFont.truetype(FONTS["bold"], 9)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    for i, (city, t) in enumerate((("ANKARA", "06:00"), ("İZMİR", "09:30"), ("SAMSUN", "13:00"), ("TRABZON", "17:30"), ("ANTALYA", "21:00"), ("ERZURUM", "23:00"))):
        d.text((146, 22 + i * 13), city, font=f, fill=(40, 40, 60)); d.text((206, 22 + i * 13), t, font=f, fill=(170, 40, 40))
    a = np.asarray(img).astype(float)
    # kapı (sol) + çerçeveler
    rect(a, 12, 30, 16, 122, (170, 170, 170)); rect(a, 112, 6, 118, 122, (170, 170, 170)); rect(a, 128, 6, 134, 122, (170, 170, 170))
    rect(a, 60, 60, 64, 76, (200, 180, 80))                           # kapı kolu
    yy, xx = np.mgrid[0:h, 0:w]
    glare = ((xx + yy * 0.6) % 70 < 12) & (xx >= 6) & (xx < 250) & (yy >= 6) & (yy < 122)
    a[glare] = a[glare] * 0.75 + np.array((170, 180, 200)) * 0.25
    return add_noise(a, 3)


def glass():
    a = solid(64, 64, (56, 66, 82))
    yy, xx = np.mgrid[0:64, 0:64]
    a[((xx + yy * 0.5) % 32 < 7)] = (96, 112, 136)
    return a


def ceil():
    a = solid(128, 128, (204, 200, 190))
    yy, xx = np.mgrid[0:128, 0:128]
    a[(xx % 64 < 3) | (yy % 64 < 3)] = (120, 118, 112)
    a += (noise(128, 128, 0.6) * 6)[..., None]
    stains(a, 4, 10, 34, (150, 140, 110), alpha=0.5, sigma=4)       # su lekeleri
    return a


def light_tube():
    a = solid(64, 16, (255, 252, 240))
    rect(a, 0, 0, 64, 2, (200, 200, 200)); rect(a, 0, 14, 64, 16, (200, 200, 200))
    rect(a, 0, 0, 3, 16, (120, 120, 120)); rect(a, 61, 0, 64, 16, (120, 120, 120))
    return a


def door():
    a = solid(64, 128, (104, 108, 112))
    add_noise(a, 5)
    rect(a, 4, 4, 60, 124, (116, 120, 124))
    rect(a, 8, 8, 56, 56, (96, 100, 104)); rect(a, 8, 64, 56, 120, (96, 100, 104))
    rect(a, 48, 58, 56, 62, (180, 170, 90))
    stains(a, 3, 4, 12, (120, 70, 36), alpha=0.5, sigma=2)
    return a


def door_wc():
    a = solid(64, 128, (190, 176, 140))
    add_noise(a, 5)
    rect(a, 0, 0, 3, 128, (120, 108, 84)); rect(a, 61, 0, 64, 128, (120, 108, 84))
    rect(a, 50, 60, 56, 66, (90, 90, 90))
    layer = text_layer(64, 128, "WC", "bold", 14, (60, 50, 40), (60, 50, 40), (18, 30), 0)
    paste_rgba(a, layer, 0.7)
    cracks(a, 3, (110, 96, 70), length=25)
    stains(a, 4, 4, 14, (130, 116, 86), alpha=0.5, sigma=2)
    return a


def door_glass():
    a = glass()
    a = np.tile(a, (2, 1, 1))
    rect(a, 0, 0, 4, 128, (160, 160, 160)); rect(a, 60, 0, 64, 128, (160, 160, 160))
    rect(a, 0, 0, 64, 4, (160, 160, 160)); rect(a, 0, 124, 64, 128, (160, 160, 160))
    rect(a, 8, 60, 20, 64, (200, 190, 120))
    return a


def mirror():
    a = solid(64, 64, (150, 160, 176))
    yy, xx = np.mgrid[0:64, 0:64]
    a[((xx - yy) % 24 < 5)] = (190, 200, 214)
    stains(a, 3, 4, 14, (110, 110, 110), alpha=0.4, sigma=2)
    cracks(a, 2, (60, 60, 70), length=40)
    rect(a, 0, 0, 64, 2, (80, 80, 80)); rect(a, 0, 62, 64, 64, (80, 80, 80))
    rect(a, 0, 0, 2, 64, (80, 80, 80)); rect(a, 62, 0, 64, 64, (80, 80, 80))
    return a


def sink():
    a = solid(64, 32, (226, 226, 220))
    yy, xx = np.mgrid[0:32, 0:64]
    a[((xx - 32) / 30.0) ** 2 + ((yy - 10) / 16.0) ** 2 > 1] = (120, 116, 106)
    rect(a, 26, 2, 38, 6, (140, 140, 140))
    stains(a, 3, 3, 8, (150, 140, 100), alpha=0.4, sigma=2)
    return add_noise(a, 3)


def urinal():
    a = solid(64, 64, (120, 116, 106))
    yy, xx = np.mgrid[0:64, 0:64]
    m = (((xx - 32) / 22.0) ** 2 + ((yy - 36) / 26.0) ** 2 < 1)
    a[m] = (226, 226, 220)
    a[m & (((xx - 32) / 14.0) ** 2 + ((yy - 40) / 18.0) ** 2 < 1)] = (200, 200, 192)
    rect(a, 28, 2, 36, 14, (150, 150, 150))
    stains(a, 3, 3, 8, (160, 150, 90), alpha=0.5, sigma=2)
    return add_noise(a, 3)


def pipe():
    a = solid(64, 64, (84, 80, 74))
    yy, xx = np.mgrid[0:64, 0:64]
    a += (np.cos((yy / 64.0) * 2 * np.pi) * 24)[..., None]
    stains(a, 4, 4, 14, (122, 70, 34), alpha=0.6, sigma=2)
    return add_noise(a, 5)


def grate():
    a = solid(64, 64, (60, 62, 64))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(yy % 8 < 3)] = (110, 112, 114)
    rect(a, 0, 0, 64, 4, (120, 120, 120)); rect(a, 0, 60, 64, 64, (120, 120, 120))
    rect(a, 0, 0, 4, 64, (120, 120, 120)); rect(a, 60, 0, 64, 64, (120, 120, 120))
    return add_noise(a, 4)


def metal():
    a = solid(64, 64, (96, 100, 104))
    return add_noise(a, 5)


def rust():
    a = solid(128, 128, (100, 104, 108))
    add_noise(a, 8)
    stains(a, 8, 8, 34, (128, 72, 36), alpha=0.7, sigma=3)
    stains(a, 5, 4, 14, (70, 40, 24), alpha=0.6, sigma=2)
    return a


# ---------------------------------------------------------------- otobüs
def bus_side(stripe=(30, 70, 150), name="ESEN TURİZM"):
    """Otobüs yanı 512x128: beyaz gövde, renkli şerit, pencere sırası, tekerlek yuvaları."""
    w, h = 512, 128
    a = solid(w, h, (232, 232, 228))
    rect(a, 0, 72, w, 84, stripe)
    rect(a, 0, 84, w, 90, tuple(c * 0.6 for c in stripe))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(9):
        x0 = 36 + i * 52
        rect(a, x0, 16, x0 + 44, 60, (40, 48, 62))
        g = ((xx + yy * 0.4) % 30 < 7) & (xx >= x0) & (xx < x0 + 44) & (yy >= 16) & (yy < 60)
        a[g] = (90, 104, 130)
    rect(a, 0, 0, w, 10, (210, 210, 206))                             # çatı kenarı
    for cx in (84, 420):                                               # tekerlek yuvaları (altta karanlık)
        a[(np.hypot(xx - cx, yy - 128) < 34)] = (30, 30, 30)
    rect(a, 0, 96, w, 100, (170, 170, 166))
    layer = text_layer(w, h, name, "bold", 22, stripe, (240, 240, 240), (200, 36), 0)
    paste_rgba(a, layer, 1.0)
    stains(a, 5, 6, 20, (150, 140, 120), alpha=0.3, sigma=3)         # çamur
    return add_noise(a, 3)


def bus_front():
    a = solid(128, 128, (226, 226, 222))
    rect(a, 10, 14, 118, 66, (40, 48, 62))
    yy, xx = np.mgrid[0:128, 0:128]
    g = ((xx + yy * 0.5) % 40 < 9) & (xx >= 10) & (xx < 118) & (yy >= 14) & (yy < 66)
    a[g] = (90, 104, 130)
    rect(a, 10, 72, 118, 84, (30, 70, 150))
    rect(a, 20, 96, 44, 110, (240, 230, 170)); rect(a, 84, 96, 108, 110, (240, 230, 170))   # farlar
    rect(a, 48, 92, 80, 116, (60, 60, 60))                                                     # ızgara
    for y in range(94, 116, 4):
        rect(a, 50, y, 78, y + 1, (110, 110, 110))
    rect(a, 0, 118, 128, 128, (50, 50, 50))
    layer = text_layer(128, 128, "ANKARA", "bold", 13, (250, 230, 60), (20, 20, 20), (30, 2), 0)
    rect(a, 20, 0, 108, 14, (20, 20, 20)); paste_rgba(a, layer, 1.0)
    return add_noise(a, 3)


def bus_back():
    a = solid(128, 128, (226, 226, 222))
    rect(a, 14, 16, 114, 60, (40, 48, 62))
    rect(a, 10, 72, 118, 84, (30, 70, 150))
    rect(a, 14, 96, 40, 112, (200, 40, 40)); rect(a, 88, 96, 114, 112, (200, 40, 40))
    rect(a, 44, 100, 84, 110, (60, 60, 60))
    rect(a, 0, 118, 128, 128, (50, 50, 50))
    return add_noise(a, 3)


def bus_roof():
    a = solid(64, 64, (214, 214, 210))
    stains(a, 3, 6, 16, (160, 156, 146), alpha=0.4, sigma=3)
    return add_noise(a, 3)


def wheel():
    a = solid(64, 64, (28, 28, 28))
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.hypot(xx - 32, yy - 32)
    a[r < 18] = (150, 150, 150)
    a[r < 6] = (90, 90, 90)
    a[(r > 24) & (r < 30) & (((np.degrees(np.arctan2(yy - 32, xx - 32)) % 30) < 12))] = (44, 44, 44)
    return a


def bay():
    """Açık bagaj bölmesi 128x64: karanlık iç, bavullar."""
    a = solid(128, 64, (26, 24, 22))
    for x0, col, wd in ((6, (120, 40, 40), 30), (40, (40, 50, 90), 26), (70, (90, 70, 40), 34), (108, (30, 30, 30), 16)):
        rect(a, x0, 24, x0 + wd, 60, col)
        rect(a, x0 + 2, 26, x0 + wd - 2, 30, tuple(c * 1.3 for c in col))
    rect(a, 0, 0, 128, 4, (120, 120, 120)); rect(a, 0, 60, 128, 64, (90, 90, 90))
    return add_noise(a, 4)


def bag(col):
    a = solid(64, 64, col)
    add_noise(a, 6)
    rect(a, 0, 0, 64, 4, tuple(c * 0.6 for c in col)); rect(a, 0, 60, 64, 64, tuple(c * 0.6 for c in col))
    rect(a, 0, 0, 4, 64, tuple(c * 0.6 for c in col)); rect(a, 60, 0, 64, 64, tuple(c * 0.6 for c in col))
    rect(a, 24, 0, 40, 8, (40, 40, 40))                              # sap
    rect(a, 8, 28, 56, 32, tuple(c * 0.7 for c in col))             # kemer
    return a


# ---------------------------------------------------------------- metro
def train():
    """M1 vagonu 512x128: beyaz, kırmızı-mavi şerit, pencereler, kapılar."""
    w, h = 512, 128
    a = solid(w, h, (224, 226, 228))
    rect(a, 0, 76, w, 86, (190, 30, 40)); rect(a, 0, 86, w, 92, (30, 60, 140))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(10):
        x0 = 16 + i * 50
        if i in (2, 7):                                                # kapılar
            rect(a, x0 - 6, 14, x0 + 50, 118, (200, 202, 204))
            rect(a, x0 + 20, 14, x0 + 24, 118, (120, 120, 120))
            rect(a, x0, 20, x0 + 18, 56, (40, 48, 62)); rect(a, x0 + 26, 20, x0 + 44, 56, (40, 48, 62))
        else:
            rect(a, x0, 20, x0 + 38, 60, (40, 48, 62))
            g = ((xx + yy * 0.4) % 30 < 6) & (xx >= x0) & (xx < x0 + 38) & (yy >= 20) & (yy < 60)
            a[g] = (90, 104, 130)
    rect(a, 0, 0, w, 8, (160, 160, 160)); rect(a, 0, 118, w, 128, (60, 60, 60))
    layer = text_layer(w, h, "M1  AKSARAY - HAVALİMANI", "bold", 12, (190, 30, 40), (240, 240, 240), (150, 98), 0)
    paste_rgba(a, layer, 1.0)
    return add_noise(a, 3)


def train_front():
    a = solid(128, 128, (224, 226, 228))
    rect(a, 16, 20, 112, 64, (40, 48, 62))
    rect(a, 0, 76, 128, 86, (190, 30, 40)); rect(a, 0, 86, 128, 92, (30, 60, 140))
    rect(a, 20, 96, 44, 108, (240, 230, 170)); rect(a, 84, 96, 108, 108, (240, 230, 170))
    rect(a, 0, 118, 128, 128, (60, 60, 60))
    layer = text_layer(128, 128, "M1", "bold", 14, (240, 240, 240), (20, 20, 20), (52, 2), 0)
    rect(a, 40, 0, 88, 16, (190, 30, 40)); paste_rgba(a, layer, 1.0)
    return add_noise(a, 3)


def rail():
    a = solid(32, 32, (110, 104, 96))
    rect(a, 0, 12, 32, 20, (160, 156, 150))
    return add_noise(a, 5)


def ballast():
    a = solid(128, 128, (70, 66, 62))
    a += (noise(128, 128, 0.6) * 22)[..., None]
    a += (noise(128, 128, 1.5) * 10)[..., None]
    return a


def tunnel():
    a = solid(128, 128, (56, 54, 52))
    add_noise(a, 8)
    stains(a, 6, 10, 40, (30, 30, 30), alpha=0.6, sigma=5)
    for x in (0, 64):
        a[:, x:x + 2] -= 14
    return a


def escal():
    a = solid(64, 64, (90, 92, 94))
    yy, xx = np.mgrid[0:64, 0:64]
    a[(xx % 4 < 2)] += 24
    rect(a, 0, 0, 64, 3, (160, 160, 160)); rect(a, 0, 61, 64, 64, (40, 40, 40))
    return add_noise(a, 3)


def turnst():
    a = solid(64, 64, (150, 152, 156))
    add_noise(a, 5)
    rect(a, 0, 0, 64, 6, (100, 102, 106)); rect(a, 0, 58, 64, 64, (100, 102, 106))
    rect(a, 20, 20, 44, 36, (40, 120, 60))                           # yeşil geçiş ışığı
    return a


def mmap():
    """Metro hat haritası afişi 128x128."""
    a = solid(128, 128, (240, 240, 236))
    rect(a, 0, 0, 128, 14, (190, 30, 40))
    layer = text_layer(128, 128, "M1", "bold", 11, (240, 240, 240), (190, 30, 40), (4, 1), 0)
    paste_rgba(a, layer, 1.0)
    rect(a, 12, 70, 116, 76, (190, 30, 40))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONTS["bold"], 7)
    for i, st in enumerate(("Aksaray", "Ulubatlı", "Bayrampaşa", "Esenler", "OTOGAR", "Yenibosna", "Havalimanı")):
        x = 14 + i * 16
        d.ellipse((x - 3, 70, x + 3, 76), fill=(255, 255, 255), outline=(190, 30, 40))
        d.text((x - 10, 80 if i % 2 == 0 else 54), st, font=f, fill=(40, 40, 60) if st != "OTOGAR" else (190, 30, 40))
    a = np.asarray(img).astype(float)
    stains(a, 2, 6, 16, (180, 170, 140), alpha=0.3, sigma=3)
    return a


# ---------------------------------------------------------------- eşyalar
def seat():
    a = solid(64, 32, (40, 80, 150))
    add_noise(a, 6)
    rect(a, 0, 0, 64, 3, (30, 60, 110)); rect(a, 31, 0, 33, 32, (20, 40, 80))
    rect(a, 0, 28, 64, 32, (80, 80, 80))
    return a


def counter():
    a = solid(64, 64, (128, 90, 52))
    add_noise(a, 6)
    rect(a, 0, 0, 64, 6, (170, 160, 140))
    for x in range(0, 64, 32):
        rect(a, x, 6, x + 1, 64, (90, 62, 36))
    return a


def shelf():
    a = solid(128, 128, (70, 72, 74))
    for y0 in (8, 48, 88):
        rect(a, 0, y0 + 32, 128, y0 + 36, (130, 132, 134))
        for i in range(4):
            x0 = 6 + i * 31
            col = (rng.uniform(140, 190), rng.uniform(110, 150), rng.uniform(70, 100))
            if rng.random() < 0.8:
                rect(a, x0, y0 + 4, x0 + 26, y0 + 32, col)
                rect(a, x0 + 2, y0 + 6, x0 + 24, y0 + 10, tuple(c * 0.8 for c in col))
    rect(a, 0, 0, 4, 128, (110, 112, 114)); rect(a, 124, 0, 128, 128, (110, 112, 114))
    return add_noise(a, 4)


def boxes():
    a = solid(64, 64, (176, 140, 92))
    add_noise(a, 6)
    rect(a, 0, 0, 64, 2, (120, 92, 56)); rect(a, 0, 32, 64, 34, (120, 92, 56))
    rect(a, 0, 0, 2, 64, (120, 92, 56)); rect(a, 32, 0, 34, 64, (120, 92, 56))
    rect(a, 6, 8, 26, 14, (90, 70, 50)); rect(a, 40, 40, 60, 46, (90, 70, 50))
    stains(a, 3, 4, 12, (120, 96, 60), alpha=0.4, sigma=2)
    return a


def sack():
    a = solid(64, 64, (150, 128, 90))
    yy, xx = np.mgrid[0:64, 0:64]
    a += (np.sin(xx * 0.9) * np.sin(yy * 0.9) * 10)[..., None]
    rect(a, 0, 0, 64, 3, (90, 76, 52)); rect(a, 0, 61, 64, 64, (90, 76, 52))
    return add_noise(a, 5)


def crate():
    a = solid(64, 64, (130, 104, 68))
    for y in range(0, 64, 16):
        a[y:y + 1] -= 40
    rect(a, 0, 0, 64, 6, (92, 70, 44)); rect(a, 0, 58, 64, 64, (92, 70, 44))
    rect(a, 0, 0, 6, 64, (92, 70, 44)); rect(a, 58, 0, 64, 64, (92, 70, 44))
    return add_noise(a, 6)


def table():
    a = solid(64, 64, (226, 222, 210))
    yy, xx = np.mgrid[0:64, 0:64]
    a[((xx // 8 + yy // 8) % 2 == 0)] = (200, 60, 60)
    rect(a, 0, 0, 64, 3, (150, 150, 150)); rect(a, 0, 61, 64, 64, (150, 150, 150))
    rect(a, 0, 0, 3, 64, (150, 150, 150)); rect(a, 61, 0, 64, 64, (150, 150, 150))
    stains(a, 3, 4, 10, (170, 140, 100), alpha=0.35, sigma=2)
    return a


def kiosk():
    a = solid(128, 128, (96, 66, 40))
    rect(a, 8, 8, 120, 120, (54, 58, 66))
    for y0 in (24, 64, 100):
        rect(a, 10, y0 + 18, 118, y0 + 22, (120, 86, 50))
        for i in range(6):
            x0 = 12 + i * 18
            col = [(220, 60, 60), (240, 200, 40), (40, 120, 200), (230, 120, 40), (120, 200, 80), (220, 220, 220)][(i + y0) % 6]
            rect(a, x0, y0, x0 + 14, y0 + 18, col)
    yy, xx = np.mgrid[0:128, 0:128]
    glare = ((xx - yy) % 56 < 8) & (xx >= 8) & (xx < 120) & (yy >= 8) & (yy < 120)
    a[glare] = a[glare] * 0.6 + np.array((150, 160, 180)) * 0.4
    return a


def tea():
    """Çay ocağı 128x64: semaver, ince belli bardaklar."""
    a = solid(128, 64, (90, 70, 50))
    add_noise(a, 5)
    rect(a, 0, 40, 128, 64, (120, 84, 48))
    yy, xx = np.mgrid[0:64, 0:128]
    sam = (((xx - 32) / 16.0) ** 2 + ((yy - 26) / 18.0) ** 2 < 1)
    a[sam] = (210, 190, 120)
    a[sam & (((xx - 32) / 9.0) ** 2 + ((yy - 24) / 10.0) ** 2 < 1)] = (236, 220, 160)
    rect(a, 28, 2, 36, 8, (180, 160, 100)); rect(a, 26, 42, 38, 46, (150, 130, 80))
    for i in range(5):
        x0 = 60 + i * 13
        rect(a, x0, 24, x0 + 9, 40, (150, 60, 30))
        rect(a, x0 + 1, 24, x0 + 8, 30, (200, 110, 60))
        rect(a, x0 - 1, 40, x0 + 10, 42, (230, 230, 230))
    return a


def barrier():
    a = solid(64, 32, (220, 40, 40))
    yy, xx = np.mgrid[0:32, 0:64]
    a[((xx + yy) // 8) % 2 == 0] = (236, 232, 220)
    rect(a, 0, 0, 64, 2, (80, 80, 80)); rect(a, 0, 30, 64, 32, (80, 80, 80))
    return add_noise(a, 4)


def fence():
    a = solid(64, 64, (0, 0, 255))
    yy, xx = np.mgrid[0:64, 0:64]
    wire = (np.abs(((xx + yy) % 16) - 8) < 1.2) | (np.abs(((xx - yy) % 16) - 8) < 1.2)
    a[wire] = (120, 124, 120)
    return a


def ladder():
    a = solid(32, 32, (0, 0, 255))
    rect(a, 0, 0, 4, 32, (90, 96, 100)); rect(a, 28, 0, 32, 32, (90, 96, 100))
    rect(a, 0, 14, 32, 18, (120, 126, 130))
    return a


def poster(title, sub, col):
    a = solid(64, 128, (240, 234, 216))
    rect(a, 4, 4, 60, 70, col)
    rect(a, 10, 40, 54, 62, (236, 236, 232)); rect(a, 12, 44, 52, 56, (40, 48, 62))     # otobüs resmi
    rect(a, 14, 60, 22, 66, (30, 30, 30)); rect(a, 42, 60, 50, 66, (30, 30, 30))
    layer = text_layer(64, 128, title, "bold", 12, (250, 250, 250), (20, 20, 20), (6, 8), 0)
    paste_rgba(a, layer, 1.0)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    d.text((6, 78), sub, font=ImageFont.truetype(FONTS["bold"], 10), fill=(60, 50, 40))
    d.text((6, 94), "HER GÜN", font=ImageFont.truetype(FONTS["bold"], 9), fill=(190, 30, 40))
    d.text((6, 106), "SEFER", font=ImageFont.truetype(FONTS["bold"], 9), fill=(190, 30, 40))
    a = np.asarray(img).astype(float)
    stains(a, 2, 4, 12, (150, 140, 110), alpha=0.3, sigma=2)
    return a


def sc_poster():
    a = solid(128, 240, (236, 232, 220))
    a[4:236, 4:124] = logo_rgb(120, 232)
    stains(a, 2, 4, 14, (150, 140, 110), alpha=0.25, sigma=2)
    return a


TEXTURES = {
    "es_asphalt": asphalt, "es_white": white, "es_yellow": yellow, "es_curb": curb,
    "es_terrazzo": terrazzo, "es_tile_brn": tile_brown, "es_tile_wet": tile_wet, "es_granite": granite,
    "es_concrete": concrete, "es_rubble": rubble,
    "es_wall": wall, "es_wall_up": wall_up, "es_wtile": wtile, "es_mtile": mtile, "es_facade": facade,
    "es_fac_low": fac_low, "es_pillar": pillar, "es_shutter": shutter, "es_shutter2": lambda: shutter(True),
    "es_shop": shop, "es_office": office, "es_glass": glass, "es_ceil": ceil, "es_light": light_tube,
    "es_door": door, "es_door_wc": door_wc, "es_door_gl": door_glass, "es_mirror": mirror, "es_sink": sink,
    "es_urinal": urinal, "es_pipe": pipe, "es_grate": grate, "es_metal": metal, "es_rust": rust,
    "es_bus1": lambda: bus_side((30, 70, 150), "ESEN TURİZM"), "es_bus2": lambda: bus_side((170, 30, 40), "ANADOLU SEYAHAT"),
    "es_bus3": lambda: bus_side((30, 120, 70), "KARADENİZ EKSPRES"), "es_bus_front": bus_front, "es_bus_back": bus_back,
    "es_bus_roof": bus_roof, "es_wheel": wheel, "es_bay": bay,
    "es_bag1": lambda: bag((120, 40, 40)), "es_bag2": lambda: bag((40, 50, 90)), "es_bag3": lambda: bag((90, 70, 40)),
    "es_train": train, "es_train_f": train_front, "es_rail": rail, "es_ballast": ballast, "es_tunnel": tunnel,
    "es_escal": escal, "es_turnst": turnst, "es_mmap": mmap,
    "es_seat": seat, "es_counter": counter, "es_shelf": shelf, "es_boxes": boxes, "es_sack": sack, "es_crate": crate,
    "es_table": table, "es_kiosk": kiosk, "es_tea": tea, "es_barrier": barrier, "{es_fence": fence, "{es_ladder": ladder,
    "es_poster1": lambda: poster("ANKARA", "6 saat", (30, 70, 150)), "es_poster2": lambda: poster("TRABZON", "17 saat", (30, 120, 70)),
    "es_sc": sc_poster,
    # tabelalar
    "es_s_otogar": lambda: sign("BÜYÜK İSTANBUL OTOGARI", 512, 64, (40, 50, 80), (240, 236, 220), "bold", 40, (20, 24, 40)),
    "es_s_metro": lambda: sign("M", 64, 64, (190, 30, 40), (250, 250, 250), "bold", 50, (120, 10, 20)),
    "es_s_metro2": lambda: sign("OTOGAR İSTASYONU", 256, 32, (190, 30, 40), (250, 250, 250), "bold", None, (120, 10, 20)),
    "es_s_wc": lambda: sign("WC", 64, 32, (40, 50, 80), (250, 250, 250), "bold"),
    "es_s_cikis": lambda: sign("ÇIKIŞ", 64, 32, (30, 110, 50), (250, 250, 250), "bold"),
    "es_s_kapali": lambda: sign("KAPALI", 64, 32, (190, 30, 40), (250, 250, 250), "bold"),
    "es_s_girilmez": lambda: sign("GİRİLMEZ", 128, 32, (190, 30, 40), (250, 250, 250), "bold"),
    "es_s_lokanta": lambda: sign("OTOGAR LOKANTASI", 256, 32, (120, 36, 36), (240, 220, 160), "serif"),
    "es_s_cay": lambda: sign("ÇAY OCAĞI", 128, 32, (90, 60, 30), (240, 220, 160), "serif"),
    "es_s_ptt": lambda: sign("PTT", 64, 32, (240, 200, 40), (40, 40, 60), "bold"),
    "es_s_polis": lambda: sign("POLİS", 128, 32, (30, 50, 120), (250, 250, 250), "bold"),
    "es_s_emanet": lambda: sign("EMANET", 128, 32, (40, 50, 80), (250, 250, 250), "bold"),
    "es_s_berber": lambda: sign("BERBER", 128, 32, (200, 40, 40), (250, 250, 250), "bold"),
    "es_s_kuafor": lambda: sign("KUAFÖR", 128, 32, (200, 80, 140), (250, 250, 250), "bold"),
    "es_s_bufe": lambda: sign("BÜFE", 128, 32, (40, 110, 50), (250, 250, 250), "bold"),
    "es_s_depo": lambda: sign("DEPO", 128, 32, (60, 60, 70), (240, 240, 240), "bold"),
    "es_s_bilet": lambda: sign("BİLET", 128, 32, (40, 50, 80), (250, 250, 250), "bold"),
    "es_s_peron": lambda: sign("PERONLAR  →", 256, 32, (40, 50, 80), (250, 250, 250), "bold"),
    "es_s_metrook": lambda: sign("← METRO", 256, 32, (190, 30, 40), (250, 250, 250), "bold"),
    "es_c1": lambda: sign("ESEN TURİZM", 256, 64, (30, 70, 150), (250, 250, 250), "bold"),
    "es_c2": lambda: sign("ANADOLU SEYAHAT", 256, 64, (170, 30, 40), (250, 240, 220), "bold"),
    "es_c3": lambda: sign("KARADENİZ EKSPRES", 256, 64, (30, 120, 70), (250, 250, 250), "bold"),
    "es_c4": lambda: sign("EGE LÜKS", 256, 64, (240, 200, 40), (40, 40, 60), "bold"),
    "es_c5": lambda: sign("DOĞU TUR", 256, 64, (60, 60, 70), (240, 200, 40), "bold"),
    "es_c6": lambda: sign("TRAKYA ÖZ", 256, 64, (120, 40, 140), (250, 250, 250), "bold"),
}
for _n in (12, 13, 14, 15, 16, 17, 18, 19):
    TEXTURES["es_p%d" % _n] = (lambda n: (lambda: sign("PERON %d" % n, 128, 32, (40, 50, 80), (250, 250, 250), "bold")))(_n)


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
    sizes = write_wad(os.path.join(here, "esenler.wad"), prev)
    print(len(sizes), "doku ->", os.path.join(here, "esenler.wad"))
