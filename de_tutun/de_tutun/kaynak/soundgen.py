"""de_tutun ortam sesleri -> cstrike/sound/tutun/*.wav (22050 Hz, 16 bit, mono).

Döngülü sesler GoldSrc'un beklediği 'cue ' parçasını (döngü başlangıcı = 0) taşır; tek seslerde
cue yoktur. Hepsi yordamsal: rüzgar, damla, metal gıcırtısı, elektrik vızıltısı, sinek, makine
uğultusu; tek sesler: uzak gök gürültüsü, metal çarpma, cam kırılması.

Kullanım: python soundgen.py [cikti_dizini]   (varsayılan: ../cstrike/sound/tutun)
"""
import os
import struct
import sys
import numpy as np

SR = 22050
rng = np.random.default_rng(1923)


def lowpass(x, cutoff):
    """Tek kutuplu alçak geçiren (basit, kararlı)."""
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    b = 1 - a
    for i in range(len(x)):
        acc = a * acc + b * x[i]
        y[i] = acc
    return y


def lowpass_fast(x, cutoff):
    """FFT ile alçak geçiren (uzun diziler için)."""
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
    """Döngü için: sonu başa çapraz karıştır."""
    n = int(fade * SR)
    head, tail = x[:n].copy(), x[-n:].copy()
    w = np.linspace(0, 1, n)
    x = x[:-n].copy()
    x[:n] = head * w + tail * (1 - w)
    return x


def normalize(x, peak=0.8):
    return x / (np.max(np.abs(x)) + 1e-9) * peak


def write_wav(path, x, loop):
    pcm = np.clip(x, -1, 1)
    data = (pcm * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, SR, SR * 2, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if loop:
        cue = struct.pack("<I", 1) + struct.pack("<II4sIII", 1, 0, b"data", 0, 0, 0)
        chunks += b"cue " + struct.pack("<I", len(cue)) + cue
    chunks += b"data" + struct.pack("<I", len(data)) + data
    open(path, "wb").write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


# ---------------------------------------------------------------- döngüler
def wind(sec=10):
    n = sec * SR
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    x = lowpass_fast(x, 600)
    gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.11 * t + 1) * np.sin(2 * np.pi * 0.047 * t)
    whistle = 0.08 * np.sin(2 * np.pi * (420 + 60 * np.sin(2 * np.pi * 0.2 * t)) * t) * (gust ** 3)
    return normalize(seamless(x * gust + whistle, 1.0), 0.7)


def drips(sec=8):
    n = sec * SR
    x = np.zeros(n)
    t0 = 0.3
    while t0 < sec - 0.5:
        f = rng.uniform(900, 2200)
        L = int(0.12 * SR)
        tt = np.arange(L) / SR
        ping = np.sin(2 * np.pi * f * tt * (1 - 0.15 * tt / 0.12)) * np.exp(-tt * 45) * rng.uniform(0.3, 1.0)
        i = int(t0 * SR)
        x[i:i + L] += ping[:n - i]
        # küçük yankı
        for d, g in ((0.09, 0.35), (0.21, 0.18)):
            j = i + int(d * SR)
            if j + L < n:
                x[j:j + L] += ping * g
        t0 += rng.uniform(0.35, 1.6)
    x += lowpass_fast(rng.standard_normal(n), 300) * 0.03          # mahzen uğultusu
    return normalize(seamless(x, 0.3), 0.75)


def creaks(sec=12):
    n = sec * SR
    x = lowpass_fast(rng.standard_normal(n), 200) * 0.04
    t0 = 0.5
    while t0 < sec - 1.5:
        L = int(rng.uniform(0.4, 1.3) * SR)
        tt = np.arange(L) / SR
        f0 = rng.uniform(90, 220)
        f = f0 * (1 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.6, 2.0) * tt))
        ph = 2 * np.pi * np.cumsum(f) / SR
        saw = ((ph / (2 * np.pi)) % 1.0) * 2 - 1
        saw = lowpass_fast(saw, 1500)
        grain = rng.standard_normal(L) * 0.25
        e = env(L, 0.08, 0.3) * rng.uniform(0.4, 1.0)
        i = int(t0 * SR)
        x[i:i + L] += ((saw * 0.7 + grain * 0.3) * e)[:n - i]
        t0 += rng.uniform(1.5, 3.5)
    return normalize(seamless(x, 0.5), 0.7)


def buzz(sec=4):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in (1, 2, 3, 5, 7):
        x += np.sin(2 * np.pi * 100 * k * t) / k
    flick = 1 + 0.25 * (lowpass_fast(rng.standard_normal(n), 8) > 0.6)
    x = x * flick + lowpass_fast(rng.standard_normal(n), 4000) * 0.05
    return normalize(seamless(x, 0.2), 0.5)


def flies(sec=6):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.zeros(n)
    for _ in range(4):
        f = rng.uniform(180, 320)
        wob = 1 + 0.12 * np.sin(2 * np.pi * rng.uniform(2, 6) * t + rng.uniform(0, 6)) + 0.05 * lowpass_fast(rng.standard_normal(n), 3)
        ph = 2 * np.pi * np.cumsum(f * wob) / SR
        amp = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * t + rng.uniform(0, 6)) ** 2
        x += (np.sign(np.sin(ph)) * 0.5 + np.sin(2 * ph) * 0.5) * amp
    x = bandpass_fast(x, 150, 2500)
    return normalize(seamless(x, 0.5), 0.6)


def hum(sec=6):
    n = sec * SR
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.25 * np.sin(2 * np.pi * 150 * t + 1)
    x *= 1 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    x += lowpass_fast(rng.standard_normal(n), 120) * 0.3
    return normalize(seamless(x, 0.5), 0.6)


# ---------------------------------------------------------------- tek sesler
def thunder(sec=4):
    n = sec * SR
    x = lowpass_fast(rng.standard_normal(n), 140)
    t = np.arange(n) / SR
    e = np.exp(-t * 1.1) * (1 + 0.6 * np.sin(2 * np.pi * 1.3 * t + 2) ** 2)
    e[:int(0.15 * SR)] *= np.linspace(0, 1, int(0.15 * SR))
    return normalize(x * e, 0.85)


def clang(sec=1.6):
    n = int(sec * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, g, dec in ((310, 1.0, 5), (742, 0.7, 7), (1180, 0.5, 9), (1990, 0.35, 12), (2630, 0.25, 15)):
        x += g * np.sin(2 * np.pi * f * t) * np.exp(-t * dec)
    x += lowpass_fast(rng.standard_normal(n), 3000) * np.exp(-t * 40) * 1.5
    return normalize(x, 0.85)


def glass(sec=1.4):
    n = int(sec * SR)
    x = np.zeros(n)
    for _ in range(14):
        i = int(rng.uniform(0, 0.5) * SR)
        L = int(0.25 * SR)
        tt = np.arange(L) / SR
        f = rng.uniform(2500, 6500)
        x[i:i + L] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * rng.uniform(15, 40)) * rng.uniform(0.3, 1)
    x += bandpass_fast(rng.standard_normal(n), 2000, 8000) * np.exp(-np.arange(n) / SR * 12) * 0.8
    return normalize(x, 0.8)


SOUNDS = {
    "ruzgar": (wind, True), "damla": (drips, True), "gicirti": (creaks, True),
    "vizilti": (buzz, True), "sinek": (flies, True), "ugultu": (hum, True),
    "gumburtu": (thunder, False), "carpma": (clang, False), "cam": (glass, False),
}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, (fn, loop) in SOUNDS.items():
        write_wav(os.path.join(outdir, name + ".wav"), fn(), loop)
        print("yazıldı", name + ".wav")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "cstrike", "sound", "tutun"))
