"""Procedural textures for de_haydarpasa, written into a WAD3 file."""
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(1908)


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
                a[r * bh:r * bh + bevel][:, xs] += 18
                a[(r + 1) * bh - bevel:(r + 1) * bh][:, xs] -= 22
                a[r * bh:(r + 1) * bh][:, xs[:bevel]] += 10
                a[r * bh:(r + 1) * bh][:, xs[-bevel:]] -= 14
            a[r * bh:(r + 1) * bh][:, xs[:mortar_w]] = mortar
        a[r * bh:r * bh + mortar_w] = mortar
    return add_noise(a, 10)


def stone():
    return blocks(128, 128, 64, 32, (196, 186, 164), (150, 142, 126), var=8)


def rustic():
    return blocks(128, 128, 64, 32, (170, 162, 146), (105, 100, 92), var=10, mortar_w=3, bevel=3)


def window(w=128, h=128, win_w=52, top=18, bottom=108):
    a = blocks(w, h, 64, 32, (196, 186, 164), (150, 142, 126), var=6)
    cx = w // 2
    x0, x1 = cx - win_w // 2, cx + win_w // 2
    # stone surround
    rect(a, x0 - 6, top - 2, x1 + 6, bottom + 6, (214, 205, 184))
    yy, xx = np.mgrid[0:h, 0:w]
    r = win_w // 2 + 6
    arch = ((xx - cx) ** 2 + (yy - (top + r)) ** 2 <= r * r) & (yy < top + r)
    a[arch] = (214, 205, 184)
    # glass
    r2 = win_w // 2
    glass = ((xx >= x0) & (xx < x1) & (yy >= top + r) & (yy < bottom)) | \
            (((xx - cx) ** 2 + (yy - (top + r)) ** 2 <= r2 * r2) & (yy < top + r) & (yy >= top + 4))
    g = np.zeros((h, w, 3))
    grad = np.clip((yy - top) / (bottom - top), 0, 1)[..., None]
    g[:] = np.array((70, 88, 104)) * (1 - grad) + np.array((38, 46, 58)) * grad
    a[glass] = g[glass]
    # white frame / muntins
    fr = (232, 228, 214)
    for xm in (x0, cx - 1, x1 - 3):
        m = glass & (xx >= xm) & (xx < xm + 3)
        a[m] = fr
    for ym in (top + r, (top + r + bottom) // 2):
        m = glass & (yy >= ym) & (yy < ym + 3)
        a[m] = fr
    # sill
    rect(a, x0 - 10, bottom, x1 + 10, bottom + 7, (226, 218, 198))
    rect(a, x0 - 10, bottom + 7, x1 + 10, bottom + 9, (120, 112, 100))
    return add_noise(a, 5)


def cornice():
    a = solid(64, 32, (218, 210, 190))
    rect(a, 0, 0, 64, 6, (232, 226, 208))
    rect(a, 0, 6, 64, 8, (150, 142, 126))
    for x in range(0, 64, 8):
        rect(a, x + 1, 10, x + 6, 20, (232, 226, 208))
        rect(a, x + 1, 20, x + 6, 22, (140, 132, 118))
    rect(a, 0, 24, 64, 26, (160, 152, 136))
    return add_noise(a, 6)


def roof():
    a = solid(128, 128, (62, 68, 78))
    for r in range(8):
        off = 8 if r % 2 else 0
        for c in range(-1, 9):
            x0 = c * 16 + off
            xs = np.arange(x0, x0 + 16) % 128
            a[r * 16:(r + 1) * 16][:, xs] += rng.uniform(-8, 8)
            a[r * 16:(r + 1) * 16][:, xs[:1]] = (34, 38, 44)
        a[r * 16 + 14:r * 16 + 16] -= 18
    return add_noise(a, 8)


def floor_marble():
    a = solid(128, 128, (220, 210, 190))
    for i in range(2):
        for j in range(2):
            if (i + j) % 2:
                rect(a, i * 64, j * 64, i * 64 + 64, j * 64 + 64, (150, 70, 52))
    veins = noise(128, 128, 3)
    a += (np.abs(veins) < 0.06)[..., None] * -25
    a[:, ::64] = (110, 100, 90)
    a[::64, :] = (110, 100, 90)
    return add_noise(a, 6)


def inwall():
    a = solid(128, 128, (226, 214, 184))
    a = add_noise(a, 6)
    rect(a, 0, 88, 128, 128, (110, 72, 42))
    for x in range(0, 128, 32):
        rect(a, x + 4, 94, x + 28, 122, (128, 86, 52))
    rect(a, 0, 86, 128, 90, (80, 50, 30))
    rect(a, 0, 0, 128, 6, (200, 186, 150))
    return a


def ceiling():
    a = solid(128, 128, (210, 198, 168))
    for i in range(2):
        for j in range(2):
            x, y = i * 64, j * 64
            rect(a, x + 6, y + 6, x + 58, y + 58, (182, 168, 136))
            rect(a, x + 12, y + 12, x + 52, y + 52, (196, 182, 148))
            rect(a, x + 28, y + 28, x + 36, y + 36, (170, 140, 70))
    return add_noise(a, 5)


def concrete():
    a = solid(128, 128, (150, 148, 142))
    a = add_noise(a, 14)
    a[:, :2] -= 30
    a[:2, :] -= 30
    return a


def cobble():
    a = solid(128, 128, (90, 88, 86))
    for r in range(8):
        off = 8 if r % 2 else 0
        for c in range(-1, 9):
            x0 = c * 16 + off
            xs = np.arange(x0 + 1, x0 + 15) % 128
            col = 118 + rng.uniform(-14, 14)
            a[r * 16 + 1:r * 16 + 15][:, xs] = (col, col - 2, col - 6)
            a[r * 16 + 1:r * 16 + 3][:, xs] += 14
    return add_noise(a, 10)


def gravel():
    a = solid(128, 128, (112, 104, 94))
    a += (noise(128, 128, 0.5) * 28)[..., None]
    a += (noise(128, 128, 3) * 12)[..., None]
    return a


def ballast():
    a = gravel()
    a *= 0.95
    for x in range(0, 128, 32):
        rect(a, x + 10, 0, x + 22, 128, (74, 56, 40))
        rect(a, x + 10, 0, x + 11, 128, (52, 40, 28))
    return add_noise(a, 5)


def rail():
    a = solid(32, 32, (96, 90, 84))
    rect(a, 0, 0, 32, 6, (176, 174, 170))
    return add_noise(a, 5)


def platform():
    a = solid(64, 64, (172, 168, 158))
    a[:, ::32] = (120, 116, 110)
    a[::32, :] = (120, 116, 110)
    return add_noise(a, 8)


def platedge():
    a = solid(32, 32, (172, 168, 158))
    rect(a, 0, 0, 32, 12, (222, 186, 40))
    return add_noise(a, 5)


def carside():
    a = solid(128, 128, (226, 226, 220))
    rect(a, 0, 104, 128, 128, (60, 60, 64))      # underframe
    rect(a, 0, 94, 128, 104, (176, 30, 36))      # red band
    rect(a, 0, 26, 128, 30, (176, 30, 36))
    for x in (8, 72):
        rect(a, x, 38, x + 48, 76, (40, 52, 66))
        rect(a, x, 38, x + 48, 41, (120, 140, 160))
    a[:, 0:1] -= 40
    return add_noise(a, 4)


def carroof():
    a = solid(64, 64, (128, 130, 132))
    a[::16, :] -= 25
    return add_noise(a, 6)


def loco():
    a = solid(128, 128, (186, 40, 36))
    rect(a, 0, 104, 128, 128, (50, 50, 54))
    rect(a, 0, 60, 128, 72, (236, 232, 222))
    for x in range(8, 128, 24):
        rect(a, x, 82, x + 14, 98, (90, 30, 28))
    return add_noise(a, 5)


def locofront():
    a = solid(64, 64, (186, 40, 36))
    rect(a, 6, 8, 58, 26, (40, 52, 66))
    rect(a, 0, 34, 64, 40, (236, 232, 222))
    rect(a, 8, 48, 16, 56, (250, 240, 180))
    rect(a, 48, 48, 56, 56, (250, 240, 180))
    return add_noise(a, 4)


def boxcar():
    a = solid(128, 128, (118, 62, 40))
    for x in range(0, 128, 8):
        a[:, x:x + 2] -= 18
    rect(a, 0, 112, 128, 128, (46, 44, 44))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    d.text((26, 40), "TCDD", fill=(235, 230, 220), font=f)
    return add_noise(np.asarray(img).astype(float), 5)


def steel():
    a = solid(64, 64, (52, 82, 68))
    return add_noise(a, 8)


def canopy():
    a = solid(128, 128, (150, 154, 156))
    for x in range(0, 128, 16):
        a[:, x:x + 4] -= 26
    return add_noise(a, 6)


def wood():
    a = solid(64, 64, (128, 96, 64))
    for y in range(0, 64, 16):
        a[y:y + 16] += rng.uniform(-14, 14)
        a[y:y + 1] -= 40
    a += (noise(64, 64, 1.0) * 10)[..., None]
    return a


def hullw():
    a = solid(128, 64, (236, 236, 230))
    yy, xx = np.mgrid[0:64, 0:128]
    for cx in (32, 96):
        m = (xx - cx) ** 2 + (yy - 28) ** 2 < 64
        a[m] = (40, 50, 62)
        ring = ((xx - cx) ** 2 + (yy - 28) ** 2 >= 64) & ((xx - cx) ** 2 + (yy - 28) ** 2 < 100)
        a[ring] = (150, 150, 146)
    rect(a, 0, 0, 128, 4, (60, 60, 60))
    return add_noise(a, 3)


def hullb():
    a = solid(64, 64, (34, 34, 38))
    rect(a, 0, 52, 64, 64, (130, 30, 30))
    return add_noise(a, 4)


def cabin():
    a = solid(64, 96, (240, 240, 234))
    rect(a, 8, 20, 56, 56, (44, 56, 70))
    rect(a, 31, 20, 33, 56, (240, 240, 234))
    rect(a, 0, 88, 64, 96, (180, 176, 168))
    return add_noise(a, 3)


def funnel():
    a = solid(64, 128, (232, 196, 90))
    rect(a, 0, 0, 64, 26, (30, 30, 32))
    return add_noise(a, 4)


def sign():
    w, h = 384, 64
    a = blocks(w, h, 64, 32, (214, 205, 184), (176, 168, 150), var=4, mortar_w=1)
    rect(a, 0, 0, w, 4, (150, 142, 126))
    rect(a, 0, h - 4, w, h, (150, 142, 126))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 40)
    text = "HAYDARPAŞA"
    bb = d.textbbox((0, 0), text, font=f)
    tx = (w - (bb[2] - bb[0])) // 2 - bb[0]
    ty = (h - (bb[3] - bb[1])) // 2 - bb[1]
    d.text((tx + 2, ty + 2), text, fill=(120, 110, 96), font=f)
    d.text((tx, ty), text, fill=(36, 40, 52), font=f)
    return np.asarray(img).astype(float)


def brick():
    return blocks(128, 128, 32, 16, (142, 64, 46), (170, 160, 146), var=14, mortar_w=2)


def crate():
    a = solid(64, 64, (150, 112, 66))
    for y in range(0, 64, 16):
        a[y:y + 1] -= 40
    rect(a, 0, 0, 64, 6, (110, 80, 46))
    rect(a, 0, 58, 64, 64, (110, 80, 46))
    rect(a, 0, 0, 6, 64, (110, 80, 46))
    rect(a, 58, 0, 64, 64, (110, 80, 46))
    yy, xx = np.mgrid[0:64, 0:64]
    a[np.abs(xx - yy) < 4] = (116, 84, 50)
    return add_noise(a, 6)


def container(rgb):
    a = solid(128, 128, rgb)
    for x in range(0, 128, 8):
        a[:, x:x + 3] -= 22
    rect(a, 0, 0, 128, 6, np.array(rgb) * 0.6)
    rect(a, 0, 122, 128, 128, np.array(rgb) * 0.6)
    return add_noise(a, 6)


def door():
    a = solid(64, 128, (196, 186, 164))
    rect(a, 6, 14, 58, 128, (88, 58, 34))
    yy, xx = np.mgrid[0:128, 0:64]
    arch = ((xx - 32) ** 2 + (yy - 40) ** 2 <= 26 * 26) & (yy < 40)
    a[arch] = (88, 58, 34)
    rect(a, 31, 30, 33, 128, (60, 38, 22))
    for y in (60, 96):
        rect(a, 10, y, 28, y + 26, (104, 70, 42))
        rect(a, 36, y, 54, y + 26, (104, 70, 42))
    return add_noise(a, 4)


def bumper():
    a = solid(32, 32, (220, 220, 220))
    yy, xx = np.mgrid[0:32, 0:32]
    a[((xx + yy) // 8) % 2 == 0] = (196, 30, 30)
    return a


def water():
    a = solid(64, 64, (34, 78, 96))
    a += (noise(64, 64, 2) * 18)[..., None]
    a += (noise(64, 64, 0.8) * 8)[..., None]
    return a


def sand():
    a = solid(128, 128, (82, 78, 62))
    return add_noise(a, 14)


def ladder():
    a = solid(32, 32, (0, 0, 255))   # 255 = transparent in "{" textures
    rect(a, 0, 0, 4, 32, (90, 96, 100))
    rect(a, 28, 0, 32, 32, (90, 96, 100))
    rect(a, 0, 14, 32, 18, (120, 126, 130))
    return a


def plaster():
    a = solid(64, 64, (214, 200, 168))
    return add_noise(a, 8)


def from_png(fname):
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    return np.asarray(Image.open(p).convert("RGB")).astype(float)


def logo():
    # iskele binası kuzey iç duvarı (160x176): sıva + çerçeveli SÇ logosu; üretimi logo_yama.py
    return from_png("hp_logo.png")


TEXTURES = {
    "hp_stone": stone, "hp_rustic": rustic, "hp_window": window,
    "hp_towerwin": lambda: window(64, 128, 30, 22, 108),
    "hp_cornice": cornice, "hp_roof": roof, "hp_floor": floor_marble,
    "hp_inwall": inwall, "hp_ceil": ceiling, "hp_concrete": concrete,
    "hp_cobble": cobble, "hp_gravel": gravel, "hp_ballast": ballast,
    "hp_rail": rail, "hp_plat": platform, "hp_platedge": platedge,
    "hp_carside": carside, "hp_carroof": carroof, "hp_loco": loco,
    "hp_locofront": locofront, "hp_boxcar": boxcar, "hp_steel": steel,
    "hp_canopy": canopy, "hp_wood": wood, "hp_hullw": hullw,
    "hp_hullb": hullb, "hp_cabin": cabin, "hp_funnel": funnel,
    "hp_sign": sign, "hp_brick": brick, "hp_crate": crate,
    "hp_contblue": lambda: container((40, 76, 128)),
    "hp_contred": lambda: container((150, 50, 38)),
    "hp_door": door, "hp_bumper": bumper, "!hp_sea": water,
    "hp_sand": sand, "{hp_ladder": ladder, "hp_plaster": plaster,
    "hp_logo": logo,
    # A'daki vagonun iki yanı, "sicakcikolata.com" yazılı (512x128); üretimi yazi_yama.py
    "hp_carsc_guney": lambda: from_png("hp_carsc_guney.png"),
    "hp_carsc_kuzey": lambda: from_png("hp_carsc_kuzey.png"),
}


def to_miptex(name, rgb):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    assert w % 16 == 0 and h % 16 == 0, name
    img = Image.fromarray(rgb, "RGB")
    if name.startswith("{"):
        # keep pure blue exact as index 255
        q = img.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = q.getpalette()[:255 * 3] + [0, 0, 255]
        arr = np.array(q)
        blue = np.all(rgb == (0, 0, 255), axis=2)
        arr[blue] = 255
        palimg = Image.new("P", (1, 1)); palimg.putpalette(pal)
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
    import os
    os.makedirs("/home/claude/hp/texprev", exist_ok=True)
    write_wad("/home/claude/hp/haydarpasa.wad", "/home/claude/hp/texprev")
    print("ok")
