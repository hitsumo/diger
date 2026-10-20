"""Esenler gündüz gökyüzü kutusu: cstrike/gfx/env/esenler{bk,dn,ft,lf,rt,up}.tga

Puslu İstanbul göğü, dağınık bulutlar, güneş (light_environment: azimut 250, yükseklik 60),
ufukta sık apartman blokları, birkaç minare, viyadük; ufkun altı gri beton.

Yön uyumu GoldSrc/Quake sky kutusuyla aynıdır: rt=+x (doğu), bk=-x (batı), lf=+y (kuzey),
ft=-y (güney), up=+z, dn=-z. Azimut: 0=doğu, 90=kuzey, 180=batı, 270=güney.

Kullanım: python skygen.py [cikti_dizini]   (varsayılan: ../cstrike/gfx/env)
"""
import os
import struct
import sys
import numpy as np

N = 256
SS = 3
NAME = "esenler"
ST_TO_VEC = [(3, -1, 2), (-3, 1, 2), (1, 3, 2), (-1, -3, 2), (-2, -1, 3), (2, -1, -3)]
SUFFIX = ["rt", "bk", "lf", "ft", "up", "dn"]
SUN_AZ, SUN_EL = 250.0, 60.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    t = np.asarray(t)[..., None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def hash2(xi, yi):
    return np.modf(np.sin(xi * 127.1 + yi * 311.7) * 43758.5453)[0] % 1.0


def vnoise(x, y):
    xi, yi = np.floor(x), np.floor(y)
    fx, fy = x - xi, y - yi
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a, b = hash2(xi, yi), hash2(xi + 1, yi)
    c, d = hash2(xi, yi + 1), hash2(xi + 1, yi + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, octaves=5):
    v, amp, tot = 0.0, 1.0, 0.0
    for _ in range(octaves):
        v = v + amp * vnoise(x, y)
        tot += amp
        x, y = x * 2.03 + 17.3, y * 2.01 + 9.1
        amp *= 0.5
    return v / tot


def hash1(i):
    return np.modf(np.sin(i * 12.9898 + 78.233) * 43758.5453)[0] % 1.0


def hills(az):
    """Uzak tepeler (derece): Ankara çanağı; kuzeydoğuda kale tepesi daha yüksek."""
    h = 0.6 + 0.5 * fbm(az * 0.12, np.zeros_like(az), 3) + 0.3 * fbm(az * 0.5 + 9, np.zeros_like(az), 2)
    return h


def castle(az, el):
    """Ankara Kalesi: tepede sur duvarı + burçlar (az 58..82)."""
    top = hills(az)
    wall = (az > 58) & (az < 82) & (el >= top - 0.2) & (el < top + 0.9)
    tow = np.zeros_like(az, dtype=bool)
    for c, w, hh in ((60, 0.9, 1.9), (66, 0.7, 1.5), (73, 1.0, 2.2), (80, 0.8, 1.7)):
        tow |= (np.abs(az - c) < w) & (el >= top - 0.2) & (el < top + hh)
    # mazgallar
    crn = (az > 58) & (az < 82) & (el >= top + 0.9) & (el < top + 1.15) & (((az * 3) % 2) < 1)
    return wall | tow | crn


def buildings(az, el):
    """Ufukta apartman/çatı dokusu, minareler, Atakule."""
    h = hills(az)
    bin_ = np.floor(az / 0.9)
    bh = h + 0.4 + 2.2 * hash1(bin_) * (hash1(bin_ + 3) > 0.15)        # sık apartmanlar
    m = (el >= h - 0.3) & (el < bh)
    win = m & (hash2(np.floor(az * 6), np.floor(el * 6)) > 0.93)      # pencere ışıltıları
    for c, hh in ((40, 3.4), (43, 3.4), (120, 3.0), (205, 2.8), (208, 2.8), (290, 3.1)):    # minareler
        m |= (np.abs(az - c) < 0.2) & (el >= h - 0.3) & (el < h + hh)
        m |= (np.abs(az - c) < 0.45) & (el >= h + hh - 0.5) & (el < h + hh - 0.1)
    # viyadük (güneyde): yatay tabliye + ayaklar
    m |= (az > 240) & (az < 275) & (np.abs(el - (h + 1.0)) < 0.15)
    for c in range(242, 275, 4):
        m |= (np.abs(az - c) < 0.15) & (el >= h - 0.3) & (el < h + 1.0)
    return m


def sky_color(d):
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    el = np.degrees(np.arcsin(np.clip(dz, -1, 1)))
    az = np.degrees(np.arctan2(dy, dx)) % 360.0
    t = np.clip(el / 90.0, 0, 1) ** 0.6
    col = lerp((208, 214, 222), (90, 130, 190), t)                   # puslu
    sr = np.radians(SUN_AZ); se = np.radians(SUN_EL)
    sun = np.array([np.cos(se) * np.cos(sr), np.cos(se) * np.sin(sr), np.sin(se)])
    cosang = np.clip(d @ sun, -1, 1)
    glow = 0.9 * np.exp((cosang - 1) * 90) + 0.35 * np.exp((cosang - 1) * 6)
    col = lerp(col, (255, 250, 232), np.clip(glow, 0, 1))
    # kümülüs: iki katman
    dzc = np.maximum(dz, 0.03)
    px, py = dx / dzc * 1.2 + 3.0, dy / dzc * 1.2 + 7.0
    n = fbm(px * 1.0, py * 1.0, 6)
    dens = smoothstep(0.50, 0.62, n) * smoothstep(1.5, 10.0, el)
    shade = fbm(px * 1.6 + 31.0, py * 1.6 + 5.0, 4)
    lit = np.clip((shade - 0.3) * 2.0 + (cosang + 1) * 0.2, 0, 1)
    cloud = lerp((170, 178, 196), (255, 255, 255), lit)
    col = lerp(col, cloud, dens * 0.96)
    n2 = fbm(px * 2.3 + 80, py * 2.3 + 40, 4)
    dens2 = smoothstep(0.56, 0.68, n2) * smoothstep(1.0, 6.0, el) * 0.6
    col = lerp(col, (240, 242, 248), dens2)
    haze = np.exp(-np.maximum(el, 0) / 8.0) * 0.65
    col = lerp(col, (214, 222, 230), np.where(el >= 0, haze, 0))
    # zemin: açık toprak/kuru çayır
    ground = lerp((150, 148, 142), (96, 94, 90), smoothstep(0, 25, -el))
    col = np.where((el < 0)[..., None], ground, col)
    # tepeler (puslu), yapılar, kale
    h = hills(az)
    hill = (el >= 0) & (el < h)
    hcol = lerp((214, 222, 230), (118, 128, 112), 0.55 + 0.3 * smoothstep(0, 4, h - el))
    col = np.where(hill[..., None], hcol, col)
    bld = buildings(az, el)
    col = np.where(bld[..., None], lerp((214, 222, 230), (150, 140, 128), 0.6), col)
    return col


def face_dirs(i, n):
    u = (np.arange(n) + 0.5) / n
    s = 2 * u - 1
    tt = 1 - 2 * u
    S, T = np.meshgrid(s, tt)
    b = (S, T, np.ones_like(S))
    v = []
    for k in ST_TO_VEC[i]:
        v.append(b[abs(k) - 1] * (1 if k > 0 else -1))
    d = np.stack(v, -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def write_tga(path, rgb):
    h, w = rgb.shape[:2]
    hdr = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, w, h, 24, 0)
    bgr = rgb[::-1, :, ::-1].astype(np.uint8).tobytes()
    open(path, "wb").write(hdr + bgr)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, suf in enumerate(SUFFIX):
        d = face_dirs(i, N * SS)
        col = sky_color(d)
        col = col.reshape(N, SS, N, SS, 3).mean(axis=(1, 3))
        write_tga(os.path.join(outdir, f"{NAME}{suf}.tga"), np.clip(col, 0, 255))
        print("yazıldı", suf)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "cstrike", "gfx", "env"))
