"""de_tutun gece gökyüzü kutusu: cstrike/gfx/env/tutun{bk,dn,ft,lf,rt,up}.tga

Koyu lacivert gök, yıldızlar, ay (light_environment yönüyle aynı: azimut 230, yükseklik 50),
ince bulutlar, güneyde şehir ışıklarının turuncu pusu, ufukta fabrika/vinç/baca siluetleri;
ufkun altı karanlık zemin.

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
NAME = "tutun"
ST_TO_VEC = [(3, -1, 2), (-3, 1, 2), (1, 3, 2), (-1, -3, 2), (-2, -1, 3), (2, -1, -3)]
SUFFIX = ["rt", "bk", "lf", "ft", "up", "dn"]
MOON_AZ, MOON_EL = 50.0, 50.0       # light_environment yaw 230: ışık güneybatıya gider, ay kuzeydoğuda


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


def skyline(az):
    """Ufuk silueti yüksekliği (derece): alçak sanayi dokusu, bacalar, vinçler, su kulesi."""
    h = 0.4 + 0.35 * fbm(az * 0.3, np.zeros_like(az), 3)
    bin_ = np.floor(az / 1.5)
    h = np.maximum(h, 0.3 + 0.9 * hash1(bin_) * (hash1(bin_ + 7) > 0.45))          # hangar/blok çatıları
    for c, w, top in ((35, 0.35, 4.2), (72, 0.3, 3.4), (128, 0.4, 5.0), (203, 0.3, 3.8), (262, 0.35, 4.5), (330, 0.3, 3.0)):
        h = np.where(np.abs(az - c) < w, top, h)                                       # bacalar
    # vinç: dikey + yatay kol
    for c, top in ((160, 3.2), (300, 2.6)):
        h = np.where(np.abs(az - c) < 0.25, top, h)
    # su kulesi
    h = np.where(np.abs(az - 95) < 0.9, 2.2 + 0.9 * np.sqrt(np.maximum(0.81 - (az - 95) ** 2, 0)), h)
    h = np.where(np.abs(az - 95) < 0.25, 2.4, np.where(np.abs(az - 95) < 0.9, np.maximum(h, 1.8), h))
    return h


def crane_arm(az, el):
    m = np.zeros_like(az, dtype=bool)
    for c, top, L in ((160, 3.2, 6.0), (300, 2.6, 5.0)):
        m |= (az > c - 0.3) & (az < c + L) & (np.abs(el - top) < 0.18)
        m |= (az > c - L * 0.3) & (az < c) & (np.abs(el - top) < 0.18)
    return m


def sky_color(d):
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    el = np.degrees(np.arcsin(np.clip(dz, -1, 1)))
    az = np.degrees(np.arctan2(dy, dx)) % 360.0
    # gece göğü: ufukta koyu mavi-gri pus, tepede siyaha yakın lacivert
    t = np.clip(el / 90.0, 0, 1) ** 0.55
    col = lerp((34, 40, 62), (6, 8, 20), t)
    # güneyde şehir ışığı pusu (turuncu), ufka yakın
    city = np.exp(-np.maximum(el, 0) / 4.0) * np.exp(-((az - 275) / 55.0) ** 2) * 0.75
    col = lerp(col, (110, 70, 40), city)
    # yıldızlar (sabit yönlere göre; ufka doğru seyrek)
    sx, sy = np.floor(az * 3.0), np.floor(el * 3.0)
    star = hash2(sx, sy)
    bright = hash2(sx + 3, sy + 5)
    starm = (star > 0.994) & (el > 2) & (hash2(sx + 9, sy + 1) < smoothstep(2, 25, el))
    col = np.where(starm[..., None], lerp(col, (200, 206, 240), 0.2 + 0.45 * bright), col)
    # ay: parlak disk + halo
    mr = np.radians(MOON_AZ); me = np.radians(MOON_EL)
    moon = np.array([np.cos(me) * np.cos(mr), np.cos(me) * np.sin(mr), np.sin(me)])
    cosang = np.clip(d @ moon, -1, 1)
    ang = np.degrees(np.arccos(cosang))
    halo = 0.35 * np.exp(-ang / 6.0) + 0.12 * np.exp(-ang / 18.0)
    col = lerp(col, (180, 190, 230), np.clip(halo, 0, 1))
    disk = smoothstep(2.3, 1.9, ang)
    col = lerp(col, (236, 238, 232), disk)
    # ince bulutlar (ay ışığında hafif aydınlık)
    dzc = np.maximum(dz, 0.03)
    px, py = dx / dzc * 1.2 + 3.0, dy / dzc * 1.2 + 7.0
    n = fbm(px * 0.8, py * 0.8, 5)
    dens = smoothstep(0.52, 0.70, n) * smoothstep(1.0, 10.0, el)
    lit = 0.35 + 0.65 * np.exp(-ang / 30.0)
    cloud = lerp((28, 32, 46), (120, 126, 150), lit)
    col = lerp(col, cloud, dens * 0.8)
    col = np.where(disk[..., None] > 0.5, lerp(col, (236, 238, 232), disk * (1 - dens)), col)
    # zemin (ufkun altı): karanlık, uzakta şehir ışığı yansıması yok
    ground = lerp((26, 26, 30), (8, 8, 10), smoothstep(0, 20, -el))
    col = np.where((el < 0)[..., None], ground, col)
    # ufuk silueti (siyaha yakın), tek tük pencere ışığı
    h = skyline(az)
    sil = (el >= 0) & (el < h)
    col = np.where(sil[..., None], (12, 12, 18), col)
    win = sil & (hash2(np.floor(az * 4), np.floor(el * 4)) > 0.975) & (el < h - 0.2)
    col = np.where(win[..., None], (200, 150, 70), col)
    col = np.where((crane_arm(az, el) & (el >= 0))[..., None], (12, 12, 18), col)
    # bacalardan hafif duman (gri, saydam)
    for c in (35, 128, 262):
        sm = np.exp(-((az - c - (el - 4) * 0.6) / 1.0) ** 2) * smoothstep(4.0, 4.5, el) * np.exp(-(el - 4) / 6.0) * 0.25
        col = lerp(col, (60, 62, 70), np.clip(sm, 0, 1))
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
