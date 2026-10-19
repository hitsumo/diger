"""de_haydarpasa.bsp: A bölgesindeki yolcu vagonunun (y=160, x 544..1024) iki yanına
pencerelerin altındaki beyaz şeride "sicakcikolata.com" yazısı.

Yeniden derleme yok: her yan için 512x128'lik yeni bir miptex eklenir, yüzlerin texinfo'su
kopyalanıp yalnız miptex değiştirilir (vecs/kaydırma aynen -> ışık haritası birebir aynı).
512, 128'in katı olduğu için arka plan eski hp_carside dokusuyla aynı fazda döşenir.

Kullanım: python yazi_yama.py girdi.bsp cikti.bsp cikti_dir
"""
import struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst, outdir = sys.argv[1:4]
TEXT = "sicakcikolata.com"
FONT = r"C:\Windows\Fonts\verdanab.ttf"
SIZE = 15                        # boxcar'daki TCDD 22 px; bu biraz daha ufak
COLOR = (92, 52, 28)             # çikolata kahvesi
STRIP = (77, 94)                 # pencere altı ile kırmızı bant arasındaki beyaz şerit (satır)
TW, TH = 512, 128

d = open(src, "rb").read()
assert struct.unpack_from("<i", d, 0)[0] == 30
L = [struct.unpack_from("<ii", d, 4 + 8 * i) for i in range(15)]
lumps = [bytearray(d[o:o + n]) for o, n in L]
tl = bytes(lumps[2])
ntex = struct.unpack_from("<i", tl, 0)[0]
toffs = list(struct.unpack_from("<%di" % ntex, tl, 4))
names = [tl[o:o + 16].split(b"\0")[0].decode() for o in toffs]
assert "hp_carsc_guney" not in names, "zaten yamalı"

o = toffs[names.index("hp_carside")]
w, h = struct.unpack_from("<II", tl, o + 16)
m0, _, _, m3 = struct.unpack_from("<4I", tl, o + 24)
idx = np.frombuffer(tl, np.uint8, w * h, o + m0).reshape(h, w)
pal = np.frombuffer(tl, np.uint8, 768, o + m3 + (w // 8) * (h // 8) + 2).reshape(256, 3)
carside = pal[idx].astype(float)
assert carside.shape == (128, 128, 3)

verts = np.frombuffer(bytes(lumps[3]), "<f4").reshape(-1, 3)
edges = np.frombuffer(bytes(lumps[12]), "<u2").reshape(-1, 2)
surf = np.frombuffer(bytes(lumps[13]), "<i4")
pl = bytes(lumps[1])
ti = lumps[6]
fl = lumps[7]


def texinfo(i):
    return np.frombuffer(bytes(ti[i * 40:i * 40 + 32]), "<f4").reshape(2, 4), struct.unpack_from("<ii", ti, i * 40 + 32)


def face(fi):
    pn, side, fe, ne, tix = struct.unpack_from("<HHiHH", fl, fi * 20)
    P = verts[[edges[surf[k]][0] if surf[k] >= 0 else edges[-surf[k]][1] for k in range(fe, fe + ne)]]
    return P, np.frombuffer(pl[pn * 20:pn * 20 + 12], "<f4") * (-1 if side else 1), tix


def to_miptex(name, rgb):   # kaynak/textures.py ile aynı biçim
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb, "RGB")
    q = img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:768]
    pal += [0] * (768 - len(pal))
    mips = [np.array(q, dtype=np.uint8)]
    for k in (2, 4, 8):
        small = img.resize((w // k, h // k), Image.Resampling.BOX)
        mips.append(np.array(small.quantize(palette=q, dither=Image.Dither.NONE), dtype=np.uint8))
    offs, data, pos = [], b"", 40
    for m in mips:
        offs.append(pos); b = m.tobytes(); data += b; pos += len(b)
    blob = name.encode()[:15].ljust(16, b"\0") + struct.pack("<II", w, h) + struct.pack("<4I", *offs) + data
    blob += struct.pack("<H", 256) + bytes(pal) + b"\0\0"
    return blob, q


SIDES = {"hp_carsc_guney": (112, -1), "hp_carsc_kuzey": (208, 1)}   # düzlem y, normal y
groups = {}
for fi in range(len(fl) // 20):
    P, nrm, tix = face(fi)
    if names[texinfo(tix)[1][0]] != "hp_carside":
        continue
    for nm, (y, ny) in SIDES.items():
        if np.allclose(P[:, 1], y) and np.allclose(nrm, (0, ny, 0)) and P[:, 0].min() >= 544 and P[:, 0].max() <= 1024:
            groups.setdefault(nm, []).append(fi)

font = ImageFont.truetype(FONT, SIZE)
new_blobs, new_ti = [], []
for k, (nm, fis) in enumerate(sorted(groups.items())):
    tixs = {face(f)[2] for f in fis}
    assert len(tixs) == 1, (nm, tixs)
    tix = tixs.pop()
    vecs, (mt, flags) = texinfo(tix)
    allP = np.vstack([face(f)[0] for f in fis])
    assert np.isclose(allP[:, 0].min(), 544) and np.isclose(allP[:, 0].max(), 1024), nm
    st = allP @ vecs[:, :3].T + vecs[:, 3]
    smin, smax = st[:, 0].min(), st[:, 0].max()
    s0 = int(round(smin))
    assert abs(smin - s0) < 0.01 and smax - smin < TW and np.isclose(st[:, 1].max() - st[:, 1].min(), 128, atol=0.01)
    fw = int(np.ceil(smax - smin))           # yüz genişliği (doku pikseli), ~452
    print(nm, "yüzler", fis, "texinfo", tix, "s %.3f..%.3f" % (smin, smax), "-> sütun kayması", s0 % TW)

    # yüz uzayında çiz (sütun p = s - smin, satır = t mod 128), sonra doku sütununa sar
    canvas = np.tile(carside, (1, TW // 128, 1))[:, :TW]
    canvas = np.roll(canvas, -(s0 % 128), axis=1)          # p -> eski doku sütunu (s0+p) mod 128
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    bb = dr.textbbox((0, 0), TEXT, font=font)
    tw_, th_ = bb[2] - bb[0], bb[3] - bb[1]
    tx = (fw - tw_) // 2 - bb[0]
    ty = (STRIP[0] + STRIP[1] - th_) // 2 - bb[1]
    # yazının arkasındaki panel ek yerlerini (128'de bir koyu çizgi) şerit içinde sil
    a = np.asarray(img).astype(float).copy()
    for p in range(tx + bb[0] - 3, tx + bb[2] + 3):
        if (s0 + p) % 128 == 0:
            a[STRIP[0]:STRIP[1], p] = a[STRIP[0]:STRIP[1], p + 1]
    img = Image.fromarray(a.astype(np.uint8))
    ImageDraw.Draw(img).text((tx, ty), TEXT, fill=COLOR, font=font)
    face_img = np.asarray(img).astype(float)
    print("   yazı", tw_, "x", th_, "doku pikseli; yüzde p=%d..%d, satır %d..%d" % (tx + bb[0], tx + bb[2], ty + bb[1], ty + bb[3]))
    assert STRIP[0] <= ty + bb[1] and ty + bb[3] <= STRIP[1]

    tex = np.roll(face_img, s0 % TW, axis=1)                # doku sütunu = (s0 + p) mod 512
    # yazı dışındaki her şey eski dokuyla aynı olmalı
    base = np.tile(carside, (1, TW // 128, 1))
    rows = np.nonzero((np.abs(tex - base).sum(2) > 0).any(1))[0]
    assert rows.min() >= STRIP[0] and rows.max() < STRIP[1], rows

    blob, q = to_miptex(nm, tex)
    q.convert("RGB").save(f"{outdir}/{nm}.png")
    open(f"{outdir}/{nm}.miptex", "wb").write(blob)
    new_blobs.append(blob)
    new_ti.append((fis, vecs.copy(), flags))

# ---------------------------------------------------------------- öbekleri yeniden kur
body = tl[4 + 4 * ntex:]
nn = ntex + len(new_blobs)
offs = [x + 4 * len(new_blobs) if x >= 0 else x for x in toffs]
data = bytearray(body)
for blob in new_blobs:
    while (4 + 4 * nn + len(data)) % 4:
        data += b"\0"
    offs.append(4 + 4 * nn + len(data))
    data += blob
lumps[2] = bytearray(struct.pack("<i", nn) + struct.pack("<%di" % nn, *offs) + data)

for k, (fis, vecs, flags) in enumerate(new_ti):
    t_new = len(ti) // 40
    ti += vecs.astype("<f4").tobytes() + struct.pack("<ii", ntex + k, flags)
    for f in fis:
        struct.pack_into("<H", fl, f * 20 + 10, t_new)
lumps[6], lumps[7] = ti, fl

order = sorted(range(15), key=lambda i: L[i][0])
out = bytearray(4 + 8 * 15)
struct.pack_into("<i", out, 0, 30)
for i in order:
    while len(out) % 4:
        out += b"\0"
    struct.pack_into("<ii", out, 4 + 8 * i, len(out), len(lumps[i]))
    out += lumps[i]
open(dst, "wb").write(out)
print("yazıldı", dst, len(out), "bayt")
