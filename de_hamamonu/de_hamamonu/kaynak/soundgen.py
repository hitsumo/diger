"""de_hamamonu ortam sesleri -> cstrike/sound/hamamonu/*.wav (22050 Hz, 16 bit, mono).

Döngüler GoldSrc'un beklediği 'cue ' parçasını taşır. Yordamsal: kuş cıvıltısı, çeşme suyu,
kalabalık uğultusu (çay bahçesi), uzak trafik, hafif rüzgar, hamamda damlayan su, lobi uğultusu;
tek sesler: korna, güvercin kanat çırpması, köpek havlaması.

Kullanım: python soundgen.py [cikti_dizini]   (varsayılan: ../cstrike/sound/hamamonu)
"""
import os
import struct
import sys
import numpy as np

SR = 22050
rng = np.random.default_rng(1071)


def lowpass_fast(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def bandpass_fast(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= (1 / (1 + (lo / np.maximum(f, 1e-6)) ** 4)) * (1 / (1 + (f / hi) ** 4))
    return np.fft.irfft(X, len(x))


def env(n, attack, release):
    e = np.ones(n)
    a = int(attack * SR); r = int(release * SR)
    e[:a] = np.linspace(0, 1, a)
    e[n - r:] = np.linspace(1, 0, r)
    return e


def seamless(x, fade=0.5):
    n = int(fade * SR)
    head, tail = x[:n].copy(), x[-n:].copy()
    w = np.linspace(0, 1, n)
    x = x[:-n].copy()
    x[:n] = head * w + tail * (1 - w)
    return x


def normalize(x, peak=0.8):
    return x / (np.max(np.abs(x)) + 1e-9) * peak


def write_wav(path, x, loop):
    data = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, SR, SR * 2, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if loop:
        cue = struct.pack("<I", 1) + struct.pack("<II4sIII", 1, 0, b"data", 0, 0, 0)
        chunks += b"cue " + struct.pack("<I", len(cue)) + cue
    chunks += b"data" + struct.pack("<I", len(data)) + data
    open(path, "wb").write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def chirp(f0, f1, dur, amp=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 + (f1 - f0) * (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env(n, 0.01, 0.03) * amp


# ---------------------------------------------------------------- döngüler
def birds(sec=14):
    n = sec * SR
    x = np.zeros(n)
    t0 = 0.4
    while t0 < sec - 1.0:
        kind = rng.integers(0, 3)
        base = rng.uniform(2200, 4200)
        cnt = int(rng.integers(2, 7))
        gap = rng.uniform(0.08, 0.2)
        for k in range(cnt):
            if kind == 0:
                c = chirp(base, base * 1.4, 0.07, rng.uniform(0.3, 0.8))
            elif kind == 1:
                c = chirp(base * 1.3, base, 0.12, rng.uniform(0.3, 0.7))
            else:
                c = chirp(base, base * 1.1, 0.05, 0.5) + np.concatenate([np.zeros(int(0.06 * SR)), chirp(base * 1.2, base * 0.9, 0.05, 0.5)])[:int(0.05 * SR)] if False else chirp(base, base * 1.6, 0.09, rng.uniform(0.3, 0.7))
            i = int((t0 + k * gap) * SR)
            if i + len(c) < n:
                x[i:i + len(c)] += c
        t0 += rng.uniform(0.9, 2.6)
    x += lowpass_fast(rng.standard_normal(n), 400) * 0.02
    return normalize(seamless(x, 0.4), 0.7)


def fountain(sec=6):
    n = sec * SR
    t = np.arange(n) / SR
    x = bandpass_fast(rng.standard_normal(n), 600, 6000)
    x *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.9 * t) + 0.1 * lowpass_fast(rng.standard_normal(n), 6)
    # su damlacıkları
    for _ in range(60):
        i = int(rng.uniform(0, sec - 0.1) * SR)
        c = chirp(rng.uniform(1500, 3500), rng.uniform(800, 1500), 0.03, 0.4)
        x[i:i + len(c)] += c
    return normalize(seamless(x, 0.5), 0.6)


def crowd(sec=12):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.zeros(n)
    # çok sayıda "konuşma" benzeri ses: formant süzgeçli, modüle edilmiş vızıltılar
    for _ in range(18):
        f0 = rng.uniform(95, 260)
        dur = rng.uniform(0.6, 2.2)
        st = rng.uniform(0, sec - dur)
        L = int(dur * SR)
        tt = np.arange(L) / SR
        f = f0 * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(2, 5) * tt) + 0.05 * lowpass_fast(rng.standard_normal(L), 10))
        ph = 2 * np.pi * np.cumsum(f) / SR
        voice = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(3, 7) * tt)))
        voice = bandpass_fast(voice, 150, 1800) * env(L, 0.1, 0.2) * rng.uniform(0.2, 0.6)
        i = int(st * SR)
        x[i:i + L] += voice
    x += bandpass_fast(rng.standard_normal(n), 200, 1500) * 0.08
    # ara sıra fincan/tabak tıkırtısı
    for _ in range(10):
        i = int(rng.uniform(0, sec - 0.2) * SR)
        c = chirp(rng.uniform(2500, 4500), rng.uniform(2000, 3000), 0.04, 0.5)
        x[i:i + len(c)] += c
    return normalize(seamless(x, 0.6), 0.6)


def traffic(sec=10):
    n = sec * SR
    t = np.arange(n) / SR
    x = lowpass_fast(rng.standard_normal(n), 220)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 1) ** 2
    # geçen araç: yükselip alçalan vızıltı
    for _ in range(3):
        st = rng.uniform(0, sec - 3)
        L = int(3 * SR)
        tt = np.arange(L) / SR
        f = rng.uniform(60, 110) * (1 + 0.15 * np.sin(np.pi * tt / 3))
        ph = 2 * np.pi * np.cumsum(f) / SR
        car = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(np.pi * tt / 3) ** 2 * 0.5
        i = int(st * SR)
        x[i:i + L] += car
    return normalize(seamless(x, 0.8), 0.55)


def wind(sec=10):
    n = sec * SR
    t = np.arange(n) / SR
    x = lowpass_fast(rng.standard_normal(n), 500)
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * t + 1) * np.sin(2 * np.pi * 0.05 * t)
    return normalize(seamless(x * gust, 1.0), 0.5)


def drips(sec=8):
    n = sec * SR
    x = np.zeros(n)
    t0 = 0.3
    while t0 < sec - 0.5:
        f = rng.uniform(900, 2000)
        L = int(0.14 * SR)
        tt = np.arange(L) / SR
        ping = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 40) * rng.uniform(0.3, 1.0)
        i = int(t0 * SR)
        x[i:i + L] += ping[:n - i]
        for d, g in ((0.11, 0.4), (0.25, 0.22), (0.4, 0.12)):     # hamam yankısı
            j = i + int(d * SR)
            if j + L < n:
                x[j:j + L] += ping * g
        t0 += rng.uniform(0.5, 1.8)
    x += bandpass_fast(rng.standard_normal(n), 300, 1200) * 0.03
    return normalize(seamless(x, 0.3), 0.7)


def hum(sec=6):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 100 * t)
    x += lowpass_fast(rng.standard_normal(n), 200) * 0.4
    return normalize(seamless(x, 0.5), 0.45)


# ---------------------------------------------------------------- tek sesler
def horn(sec=0.9):
    n = int(sec * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in (392, 494):
        x += np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f * t) * 0.5
    x = bandpass_fast(x, 200, 3500) * env(n, 0.02, 0.1)
    x[int(0.42 * SR):int(0.48 * SR)] *= np.linspace(1, 0.1, int(0.06 * SR))
    return normalize(x, 0.8)


def pigeons(sec=1.8):
    n = int(sec * SR)
    x = np.zeros(n)
    t0 = 0.0
    while t0 < sec - 0.1:
        L = int(0.05 * SR)
        burst = bandpass_fast(rng.standard_normal(L), 300, 2500) * env(L, 0.005, 0.03)
        i = int(t0 * SR)
        x[i:i + L] += burst * rng.uniform(0.5, 1.0)
        t0 += rng.uniform(0.07, 0.12)
    # "guk guk"
    for st in (0.1, 0.35):
        c = chirp(420, 380, 0.18, 0.6)
        i = int(st * SR)
        x[i:i + len(c)] += lowpass_fast(c, 1200)
    return normalize(x, 0.75)


def dog(sec=1.6):
    n = int(sec * SR)
    x = np.zeros(n)
    for st in (0.1, 0.55, 1.0):
        L = int(0.22 * SR)
        tt = np.arange(L) / SR
        f = 180 * (1 + 0.6 * np.exp(-tt * 25))
        ph = 2 * np.pi * np.cumsum(f) / SR
        bark = np.sign(np.sin(ph)) * 0.5 + np.sin(ph) * 0.5
        bark = bandpass_fast(bark + rng.standard_normal(L) * 0.3, 250, 2200) * env(L, 0.01, 0.12)
        i = int(st * SR)
        x[i:i + L] += bark
    return normalize(x, 0.8)


SOUNDS = {
    "kus": (birds, True), "cesme": (fountain, True), "kalabalik": (crowd, True), "trafik": (traffic, True),
    "ruzgar": (wind, True), "damla": (drips, True), "ugultu": (hum, True),
    "korna": (horn, False), "guvercin": (pigeons, False), "kopek": (dog, False),
}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, (fn, loop) in SOUNDS.items():
        write_wav(os.path.join(outdir, name + ".wav"), fn(), loop)
        print("yazıldı", name + ".wav")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "cstrike", "sound", "hamamonu"))
