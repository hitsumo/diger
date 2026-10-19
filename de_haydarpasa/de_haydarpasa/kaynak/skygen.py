"""de_haydarpasa için gökyüzü kutusu (skybox) üretir: cstrike/gfx/env/haydarpasa{bk,dn,ft,lf,rt,up}.tga

Bulutlu mavi gök, ufukta puslu İstanbul silueti (tarihi yarımada: kubbeler, minareler, Galata
kulesi), kuzeybatıda Boğaz Köprüsü silueti; ufkun altı denizin devamı (haritadaki !hp_sea ile
uyumlu) ki deniz kutunun kenarında kesilmeden ufka kadar sürsün.

Yön uyumu GoldSrc/Quake sky kutusuyla aynıdır (st_to_vec tablosu): rt=+x (doğu), bk=-x (batı,
deniz), lf=+y (kuzey), ft=-y (güney), up=+z, dn=-z. Azimut: 0=doğu, 90=kuzey, 180=batı, 270=güney.

Kullanım: python skygen.py [cikti_dizini]   (varsayılan: ../cstrike/gfx/env)
"""
import os
import struct
import sys
import numpy as np

N = 256          # yüz boyutu (CS 1.6 için güvenli)
SS = 3           # kenar yumuşatma için üst örnekleme
NAME = "haydarpasa"

# Quake/GoldSrc: yüz sırası rt, bk, lf, ft, up, dn ; vec[j] = ±b[|k|-1], b = (s, t, 1)
ST_TO_VEC = [(3, -1, 2), (-3, 1, 2), (1, 3, 2), (-1, -3, 2), (-2, -1, 3), (2, -1, -3)]
SUFFIX = ["rt", "bk", "lf", "ft", "up", "dn"]


# ---------------------------------------------------------------- yardımcılar
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
    for i in range(octaves):
        v = v + amp * vnoise(x, y)
        tot += amp
        x, y = x * 2.03 + 17.3, y * 2.01 + 9.1
        amp *= 0.5
    return v / tot


def hash1(i):
    return np.modf(np.sin(i * 12.9898 + 78.233) * 43758.5453)[0] % 1.0


# ---------------------------------------------------------------- ufuk silueti
def skyline(az):
    """Azimuta göre kıyı siluetinin üst sınırı (derece) ve tür: 0 yok, 1 tepe, 2 yapı, 3 köprü."""
    h = np.zeros_like(az)
    kind = np.zeros_like(az, dtype=np.int8)
    # uzak tepeler: batı yarısı (deniz tarafı), uçlarda sönümlenir
    base = 0.0
    for c, w, a in ((150, 10, 1.2), (185, 14, 1.4), (210, 12, 1.0), (240, 16, 0.6), (115, 10, 0.6), (100, 8, 0.4)):
        base = base + a * np.exp(-((az - c) / w) ** 2)
    base = base + 0.35 * fbm(az * 0.35, np.zeros_like(az), 3)
    taper = smoothstep(92, 104, az) * (1 - smoothstep(258, 270, az))
    hills = np.maximum(base * taper, 0)
    m = hills > 0.02
    h[m] = hills[m]
    kind[m] = 1
    # şehir dokusu: küçük yapı dikdörtgenleri (az 125..255)
    bin_ = np.floor(az / 0.45)
    bh = 0.35 + 0.6 * hash1(bin_)
    town = smoothstep(125, 135, az) * (1 - smoothstep(250, 258, az))
    bh = bh * town + hills * 0.6
    m = bh > h
    h[m] = bh[m]
    kind[m] = 2
    # kubbeler + minareler (tarihi yarımada)
    for c, r, hb, mins in ((186, 1.7, 1.4, (183.1, 188.9)), (195, 1.3, 1.3, (192.4, 197.6, 193.4, 196.6)),
                           (175, 1.0, 1.2, (173.3, 176.7))):
        d = np.sqrt(np.maximum(r * r - (az - c) ** 2, 0))
        dome = np.where(np.abs(az - c) < r, hb + d, 0)
        m = dome > h
        h[m] = dome[m]
        kind[m] = 2
        for mc in mins:
            spike = np.where(np.abs(az - mc) < 0.2, hb + 2.0, 0)
            m = spike > h
            h[m] = spike[m]
            kind[m] = 2
    # Galata kulesi
    tower = np.where(np.abs(az - 150) < 0.36, 2.6 - np.abs(az - 150) * 1.2, 0)
    tower = np.where(np.abs(az - 150) < 0.26, 2.6, tower)
    m = tower > h
    h[m] = tower[m]
    kind[m] = 2
    return h, kind


def bridge_mask(az, el):
    """Boğaz Köprüsü silueti: asma köprü, az 112..142, kuleler 4° yüksek."""
    a0, a1 = 112.0, 142.0
    t0, t1 = a0 + 7.5, a1 - 7.5
    top, deck = 5.2, 0.7
    m = np.zeros_like(az, dtype=bool)
    inside = (az >= a0) & (az <= a1)
    # tabliye
    m |= inside & (el >= deck) & (el <= deck + 0.6)
    # kuleler (H biçimi: iki ayak + iki kiriş)
    for tc in (t0, t1):
        legs = (np.abs(np.abs(az - tc) - 0.5) < 0.24) & (el >= 0) & (el <= top)
        beams = (np.abs(az - tc) < 0.75) & (((el > top - 0.6) & (el <= top)) | (np.abs(el - 2.6) < 0.3))
        m |= legs | beams
    # ana kablo: kule tepelerinden ortada tabliyeye inen parabol
    mid = (t0 + t1) / 2
    half = (t1 - t0) / 2
    cable = deck + 0.7 + (top - deck - 0.7) * ((az - mid) / half) ** 2
    main = (az >= t0) & (az <= t1)
    m |= main & (np.abs(el - cable) < 0.26)
    # yan kablolar: kule tepesinden tabliye ucuna
    for tc, end in ((t0, a0), (t1, a1)):
        s = (az - tc) / (end - tc)
        side = top - (top - deck) * np.clip(s, 0, 1)
        m |= (np.minimum(tc, end) <= az) & (az <= np.maximum(tc, end)) & (np.abs(el - side) < 0.26)
    # askılar
    hang = main & (np.abs(((az - t0) % 1.25) - 0.625) < 0.11) & (el > deck) & (el < cable)
    m |= hang
    return m


# ---------------------------------------------------------------- renk
SUN_AZ, SUN_EL = 20.0, 40.0      # light_environment: yaw 20, pitch -40


def sky_color(d):
    """d: (..., 3) birim yön vektörleri -> (..., 3) RGB (0..255)."""
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    el = np.degrees(np.arcsin(np.clip(dz, -1, 1)))
    az = np.degrees(np.arctan2(dy, dx)) % 360.0

    # gök gradyanı
    t = np.clip(el / 90.0, 0, 1) ** 0.6
    col = lerp((200, 216, 234), (58, 116, 200), t)
    # güneş parıltısı
    sr = np.radians(SUN_AZ); se = np.radians(SUN_EL)
    sun = np.array([np.cos(se) * np.cos(sr), np.cos(se) * np.sin(sr), np.sin(se)])
    cosang = np.clip(d @ sun, -1, 1)
    glow = 0.85 * np.exp((cosang - 1) * 60) + 0.3 * np.exp((cosang - 1) * 5)
    col = lerp(col, (255, 246, 222), np.clip(glow, 0, 1))
    # bulutlar: sabit yükseklikteki katmana izdüşüm (ufka doğru sıklaşır)
    dzc = np.maximum(dz, 0.03)
    px, py = dx / dzc * 1.2 + 3.0, dy / dzc * 1.2 + 7.0
    n = fbm(px * 1.1, py * 1.1, 6)
    dens = smoothstep(0.50, 0.64, n) * smoothstep(1.0, 9.0, el)
    shade = fbm(px * 1.4 + 31.0, py * 1.4 + 5.0, 4)
    lit = np.clip((shade - 0.35) * 2.2 + (cosang + 1) * 0.15, 0, 1)
    cloud = lerp((176, 186, 204), (255, 255, 255), lit)
    col = lerp(col, cloud, dens * 0.95)
    # ufuk pusu
    haze = np.exp(-np.maximum(el, 0) / 5.0) * 0.55
    col = lerp(col, (214, 225, 237), np.where(el >= 0, haze, 0))

    # deniz (ufkun altı): uzakta puslu açık, aşağıda haritadaki su rengine yaklaşır
    sea = lerp((112, 158, 182), (34, 80, 100), smoothstep(0, 30, -el))
    col = np.where((el < 0)[..., None], sea, col)

    # kıyı silueti
    h, kind = skyline(az)
    hill_col = lerp((214, 225, 237), (128, 150, 180), 0.75)
    bld_col = lerp((214, 225, 237), (112, 132, 162), 0.8)
    sil = (el >= 0) & (el < h)
    col = np.where((sil & (kind == 1))[..., None], hill_col, col)
    col = np.where((sil & (kind == 2))[..., None], bld_col, col)
    # köprü
    br = bridge_mask(az, el) & (el >= 0)
    col = np.where(br[..., None], lerp((214, 225, 237), (96, 112, 142), 0.8), col)
    return col


# ---------------------------------------------------------------- yüzler
def face_dirs(i, n):
    u = (np.arange(n) + 0.5) / n
    s = 2 * u - 1
    tt = 1 - 2 * u
    S, T = np.meshgrid(s, tt)            # satır = t (üstten alta), sütun = s
    b = (S, T, np.ones_like(S))
    v = []
    for k in ST_TO_VEC[i]:
        v.append(b[abs(k) - 1] * (1 if k > 0 else -1))
    d = np.stack(v, -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def write_tga(path, rgb):
    h, w = rgb.shape[:2]
    hdr = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, w, h, 24, 0)   # sol alt köşe başlangıç
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
