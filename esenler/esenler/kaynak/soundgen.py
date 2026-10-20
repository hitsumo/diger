"""Esenler Otogarı ortam sesleri -> cstrike/sound/esenler/*.wav (22050 Hz, 16 bit, mono).

Döngüler GoldSrc'un beklediği 'cue ' parçasını taşır. Yordamsal: otobüs rölantisi, kalabalık,
floresan vızıltısı, metro tüneli uğultusu, rüzgar, havalandırma, damla; tek sesler: anons
(ding-dong + boğuk anons), çığırtkan ("Ankara, Ankara!" formant sentezi, uzaktan), bagaj kapağı,
hava freni, metro gelişi.

Kullanım: python soundgen.py [cikti_dizini]   (varsayılan: ../cstrike/sound/esenler)
"""
import os
import struct
import sys
import numpy as np

SR = 22050
rng = np.random.default_rng(1994)


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


def reverb(x, delays=(0.031, 0.067, 0.113, 0.191), gains=(0.5, 0.35, 0.25, 0.15), wet=0.5):
    y = x.copy()
    for d, g in zip(delays, gains):
        k = int(d * SR)
        if k < len(x):
            y[k:] += x[:-k] * g
    return x * (1 - wet) + y * wet


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
def bus_idle(sec=8):
    n = sec * SR
    t = np.arange(n) / SR
    f = 24.0 * (1 + 0.02 * np.sin(2 * np.pi * 0.3 * t) + 0.01 * lowpass_fast(rng.standard_normal(n), 3))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.zeros(n)
    for k, g in ((1, 1.0), (2, 0.7), (3, 0.5), (4, 0.4), (6, 0.25), (8, 0.15)):
        x += g * np.sin(k * ph + 0.3 * k)
    x += lowpass_fast(rng.standard_normal(n), 300) * 0.5           # dizel hışırtısı
    x = lowpass_fast(x, 500)
    return normalize(seamless(x, 0.6), 0.6)


def crowd(sec=12):
    n = sec * SR
    x = np.zeros(n)
    for _ in range(24):
        f0 = rng.uniform(95, 260)
        dur = rng.uniform(0.5, 2.0)
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
    for _ in range(8):                                               # bavul tekerleği / ayak sesi
        i = int(rng.uniform(0, sec - 0.3) * SR)
        L = int(0.25 * SR)
        x[i:i + L] += lowpass_fast(rng.standard_normal(L), 400) * env(L, 0.02, 0.1) * 0.3
    return normalize(seamless(x, 0.6), 0.6)


def fluorescent(sec=4):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in (1, 2, 3, 5, 7):
        x += np.sin(2 * np.pi * 100 * k * t) / k
    flick = 1 + 0.3 * (lowpass_fast(rng.standard_normal(n), 8) > 0.6)
    x = x * flick + lowpass_fast(rng.standard_normal(n), 4000) * 0.05
    return normalize(seamless(x, 0.2), 0.45)


def metro_tunnel(sec=10):
    n = sec * SR
    t = np.arange(n) / SR
    x = lowpass_fast(rng.standard_normal(n), 160)
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t + 1) * np.sin(2 * np.pi * 0.05 * t)
    x = x * gust + np.sin(2 * np.pi * 50 * t) * 0.15                 # trafo uğultusu
    return normalize(seamless(x, 1.0), 0.5)


def wind(sec=10):
    n = sec * SR
    t = np.arange(n) / SR
    x = lowpass_fast(rng.standard_normal(n), 500)
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * t + 1) * np.sin(2 * np.pi * 0.05 * t)
    return normalize(seamless(x * gust, 1.0), 0.45)


def ventilation(sec=6):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 180 * t)
    x += bandpass_fast(rng.standard_normal(n), 200, 900) * 0.6
    x *= 1 + 0.08 * np.sin(2 * np.pi * 1.3 * t)
    return normalize(seamless(x, 0.5), 0.45)


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
        for d, g in ((0.09, 0.4), (0.21, 0.2)):
            j = i + int(d * SR)
            if j + L < n:
                x[j:j + L] += ping * g
        t0 += rng.uniform(0.5, 1.8)
    x += bandpass_fast(rng.standard_normal(n), 300, 1200) * 0.03
    return normalize(seamless(x, 0.3), 0.7)


# ---------------------------------------------------------------- tek sesler
def announce(sec=4.0):
    n = int(sec * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for st, f in ((0.0, 659), (0.45, 523)):                           # ding-dong
        L = int(0.9 * SR)
        tt = np.arange(L) / SR
        tone = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt * 3.5)
        i = int(st * SR)
        x[i:i + L] += tone * 0.6
    # boğuk anons: formant benzeri modüle edilmiş uğultu
    st = 1.3
    L = int(2.4 * SR)
    tt = np.arange(L) / SR
    f0 = 150 * (1 + 0.1 * np.sin(2 * np.pi * 1.7 * tt) + 0.06 * lowpass_fast(rng.standard_normal(L), 6))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    v = (np.sign(np.sin(ph)) * 0.4 + np.sin(ph) * 0.6) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.5 * tt)) ** 2)
    v = bandpass_fast(v, 300, 1600) * env(L, 0.05, 0.3)
    i = int(st * SR)
    x[i:i + L] += v * 0.5
    x = reverb(x, wet=0.6)
    return normalize(x, 0.7)


def formant_speech(segs, f0_start, f0_end, shout=True):
    """Basit formant sentezi. segs: (sure, F1, F2, F3, genlik, tur) tur: 'v' sesli, 'n' nazal, 'k' patlamalı, 'r' vuruşlu."""
    total = sum(s[0] for s in segs)
    n = int(total * SR)
    t = np.arange(n) / SR
    f0 = f0_start + (f0_end - f0_start) * (t / total) + 4 * np.sin(2 * np.pi * 6 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    src = (ph % (2 * np.pi)) / (2 * np.pi) * 2 - 1                      # testere (glottal)
    src = src - lowpass_fast(src, 80)
    out = np.zeros(n)
    pos = 0
    frame = int(0.012 * SR)
    hop = frame // 2
    win = np.hanning(frame)
    # segment başına formant hedefleri -> zaman dizisi (yumuşatılmış)
    F = np.zeros((n, 3)); A = np.zeros(n); kind = np.empty(n, dtype="<U1")
    for dur, f1, f2, f3, amp, tp in segs:
        L = int(dur * SR)
        F[pos:pos + L] = (f1, f2, f3); A[pos:pos + L] = amp; kind[pos:pos + L] = tp
        pos += L
    for c in range(3):
        F[:, c] = lowpass_fast(F[:, c], 40)
    A = lowpass_fast(A, 40)
    noise = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    for i in range(0, n - frame, hop):
        seg = src[i:i + frame] * win
        k = kind[i + frame // 2]
        if k == "k":
            seg = noise[i:i + frame] * win * 0.8
        elif k == "n":
            seg = seg * 0.5
        S = np.fft.rfft(seg)
        f1, f2, f3 = F[i + frame // 2]
        H = np.zeros_like(freqs)
        for fc, bw, g in ((f1, 90, 1.0), (f2, 110, 0.7), (f3, 160, 0.35), (3500, 250, 0.12)):
            H += g / (1 + ((freqs - fc) / bw) ** 2)
        if k == "k":
            H = 0.3 + 1 / (1 + ((freqs - 2500) / 800) ** 2)
        out[i:i + frame] += np.fft.irfft(S * H, frame) * A[i + frame // 2]
    return out


def tout(word_segs, reps=2, gap=0.35, f0=(230, 170)):
    parts = []
    for r in range(reps):
        parts.append(formant_speech(word_segs, f0[0] * (1 - 0.05 * r), f0[1]))
        parts.append(np.zeros(int(gap * SR)))
    x = np.concatenate(parts)
    x = bandpass_fast(x, 180, 3200)
    x = reverb(x, delays=(0.041, 0.083, 0.149, 0.233), gains=(0.5, 0.4, 0.3, 0.2), wet=0.55)
    return normalize(x, 0.7)


ANKARA = [(0.22, 730, 1090, 2440, 1.0, "v"), (0.07, 300, 1200, 2500, 0.6, "n"), (0.05, 300, 1200, 2500, 0.0, "k"), (0.05, 0, 0, 0, 0.7, "k"),
          (0.22, 730, 1090, 2440, 1.0, "v"), (0.05, 450, 1300, 1700, 0.5, "r"), (0.38, 730, 1090, 2440, 1.0, "v")]
SAMSUN = [(0.06, 0, 0, 0, 0.5, "k"), (0.20, 730, 1090, 2440, 1.0, "v"), (0.08, 300, 1200, 2500, 0.6, "n"), (0.06, 0, 0, 0, 0.5, "k"),
          (0.30, 300, 870, 2240, 1.0, "v"), (0.12, 300, 1200, 2500, 0.6, "n")]


def luggage(sec=1.2):
    n = int(sec * SR)
    t = np.arange(n) / SR
    x = lowpass_fast(rng.standard_normal(n), 220) * np.exp(-t * 14) * 1.5
    x += np.sin(2 * np.pi * 90 * t) * np.exp(-t * 9)
    j = int(0.5 * SR)
    x[j:] += (lowpass_fast(rng.standard_normal(n - j), 400) * np.exp(-t[:n - j] * 20)) * 0.8        # kapak kapanışı
    return normalize(x, 0.8)


def air_brake(sec=1.6):
    n = int(sec * SR)
    t = np.arange(n) / SR
    x = bandpass_fast(rng.standard_normal(n), 1500, 7000) * np.exp(-t * 3)
    x[:int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))
    x += lowpass_fast(rng.standard_normal(n), 300) * np.exp(-t * 30) * 0.8
    return normalize(x, 0.75)


def metro_arrive(sec=6.0):
    n = int(sec * SR)
    t = np.arange(n) / SR
    rumble = lowpass_fast(rng.standard_normal(n), 120) * (np.sin(np.pi * t / sec) ** 0.5)
    squeal = np.sin(2 * np.pi * (2400 + 300 * np.sin(2 * np.pi * 3 * t)) * t) * np.exp(-((t - 3.6) / 0.7) ** 2) * 0.25
    hiss = bandpass_fast(rng.standard_normal(n), 1000, 6000) * np.exp(-((t - 4.6) / 0.25) ** 2) * 0.6
    x = rumble + squeal + hiss
    x = reverb(x, wet=0.4)
    return normalize(x, 0.8)


SOUNDS = {
    "otobus": (bus_idle, True), "kalabalik": (crowd, True), "floresan": (fluorescent, True), "metro": (metro_tunnel, True),
    "ruzgar": (wind, True), "havalandirma": (ventilation, True), "damla": (drips, True),
    "anons": (announce, False), "cigirtkan": (lambda: tout(ANKARA, 2), False), "cigirtkan2": (lambda: tout(SAMSUN, 2, f0=(200, 160)), False),
    "bagaj": (luggage, False), "fren": (air_brake, False), "metro_gelis": (metro_arrive, False),
}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, (fn, loop) in SOUNDS.items():
        write_wav(os.path.join(outdir, name + ".wav"), fn(), loop)
        print("yazıldı", name + ".wav")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "cstrike", "sound", "esenler"))
