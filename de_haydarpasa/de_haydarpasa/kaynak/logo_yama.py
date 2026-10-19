"""de_haydarpasa.bsp: iskele kulübesinin kuzey iç duvarına (yüz y=144, -y'ye bakan) logo.

Haritayı yeniden derlemeden: yeni bir miptex (hp_logo, 128x256) BSP doku öbeğine eklenir,
duvar yüzünün texinfo'su kopyalanıp yalnız miptex ve t kaydırması (48 -> 176, 16'nın katı:
ışık haritası ızgarası birebir aynı kalır) değiştirilir.

Kullanım: python logo_yama.py girdi.bsp logo.png cikti.bsp cikti_dir
"""
import struct, sys
import numpy as np
from PIL import Image, ImageFilter

src, logo_path, dst, outdir = sys.argv[1:5]
d = open(src, "rb").read()
assert struct.unpack_from("<i", d, 0)[0] == 30
L = [list(struct.unpack_from("<ii", d, 4 + 8 * i)) for i in range(15)]
lumps = [bytearray(d[o:o + n]) for o, n in L]

# ---------------------------------------------------------------- dokular
tl = bytes(lumps[2])
ntex = struct.unpack_from("<i", tl, 0)[0]
toffs = list(struct.unpack_from("<%di" % ntex, tl, 4))
names = [tl[o:o + 16].split(b"\0")[0].decode() for o in toffs]
assert "hp_logo" not in names, "zaten yamalı"


def miptex_rgb(name):
    o = toffs[names.index(name)]
    w, h = struct.unpack_from("<II", tl, o + 16)
    m0, _, _, m3 = struct.unpack_from("<4I", tl, o + 24)
    idx = np.frombuffer(tl, np.uint8, w * h, o + m0).reshape(h, w)
    po = o + m3 + (w // 8) * (h // 8) + 2
    pal = np.frombuffer(tl, np.uint8, 768, po).reshape(256, 3)
    return pal[idx].astype(float)


# ---------------------------------------------------------------- hedef yüz
verts = np.frombuffer(bytes(lumps[3]), "<f4").reshape(-1, 3)
edges = np.frombuffer(bytes(lumps[12]), "<u2").reshape(-1, 2)
surf = np.frombuffer(bytes(lumps[13]), "<i4")
pl = bytes(lumps[1])
ti = lumps[6]
fl = lumps[7]


def texinfo(i):
    return np.frombuffer(bytes(ti[i * 40:i * 40 + 32]), "<f4").reshape(2, 4).copy(), struct.unpack_from("<ii", ti, i * 40 + 32)


def face_verts(fi):
    _, _, fe, ne, _ = struct.unpack_from("<HHiHH", fl, fi * 20)
    return verts[[edges[surf[k]][0] if surf[k] >= 0 else edges[-surf[k]][1] for k in range(fe, fe + ne)]]


targets = []
for fi in range(len(fl) // 20):
    pn, side, fe, ne, tix = struct.unpack_from("<HHiHH", fl, fi * 20)
    nrm = np.frombuffer(pl[pn * 20:pn * 20 + 12], "<f4") * (-1 if side else 1)
    if names[texinfo(tix)[1][0]] != "hp_plaster" or not np.allclose(nrm, (0, -1, 0)):
        continue
    P = face_verts(fi)
    if np.allclose(P[:, 1], 144) and P[:, 0].min() >= -1584 and P[:, 0].max() <= -1424:
        targets.append(fi)
assert len(targets) == 1, targets
FACE = targets[0]
P = face_verts(FACE)
_, _, _, _, old_tix = struct.unpack_from("<HHiHH", fl, FACE * 20)
vecs, (old_mt, flags) = texinfo(old_tix)
print("yüz", FACE, "texinfo", old_tix, vecs.tolist(), "x", P[:, 0].min(), P[:, 0].max(), "z", P[:, 2].min(), P[:, 2].max())
assert np.allclose(vecs[:, :3], [[1, 0, 0], [0, 0, -1]]) and np.allclose(vecs[:, 3], [48, 48])

TW, TH = 128, 256
new_vecs = vecs.copy()
new_vecs[1, 3] = 176.0            # t = 176 - z  -> satır 0 = duvar tepesi (z=176)
# s = x + 48 = (x + 1584) - 1536 -> sütun = duvar içi x (0..159) mod 128


def extents(v):
    st = P @ v[:, :3].T + v[:, 3]
    mn, mx = np.floor(st.min(0) / 16), np.ceil(st.max(0) / 16)
    return mn, mx - mn


(m0, e0), (m1, e1) = extents(vecs), extents(new_vecs)
assert np.array_equal(e0, e1) and np.array_equal((m1 - m0) * 16, new_vecs[:, 3] - vecs[:, 3]), (m0, e0, m1, e1)

# ---------------------------------------------------------------- doku tasarımı
plaster = miptex_rgb("hp_plaster")                    # 64x64; eski eşlemeyle aynı faz
tex = np.tile(plaster, (TH // 64, TW // 64, 1))       # 256x128

LW, LH, FR = 76, 142, 4                               # logo + çerçeve
OW, OH = LW + 2 * FR, LH + 2 * FR                     # 84 x 150
cx, top = 80, 10                                      # duvar ortası; tepe z=166, alt z=16
x0, y0 = cx - OW // 2, top

logo = Image.open(logo_path).convert("RGB").resize((LW, LH), Image.Resampling.LANCZOS)

img = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))
# yumuşak gölge (sağ-alt)
sh = Image.new("L", (TW, TH), 0)
sh.paste(150, (x0 + 2, y0 + 3, x0 + OW + 2, y0 + OH + 3))
sh = sh.filter(ImageFilter.GaussianBlur(1.5))
a = np.asarray(img).astype(float)
k = np.asarray(sh).astype(float)[..., None] / 255.0
a = a * (1 - 0.55 * k)
# koyu ahşap çerçeve + iç/dış kenar ışığı
wood = np.array((62, 40, 24), float)
a[y0:y0 + OH, x0:x0 + OW] = wood
a[y0, x0:x0 + OW] += 30; a[y0:y0 + OH, x0] += 22                      # dış üst/sol ışık
a[y0 + OH - 1, x0:x0 + OW] -= 18; a[y0:y0 + OH, x0 + OW - 1] -= 14     # dış alt/sağ gölge
a[y0 + FR - 1, x0 + FR - 1:x0 + OW - FR + 1] -= 20                     # iç kenar
a[y0 + FR - 1:y0 + OH - FR + 1, x0 + FR - 1] -= 20
a[y0 + OH - FR, x0 + FR - 1:x0 + OW - FR + 1] += 18
a[y0 + FR - 1:y0 + OH - FR + 1, x0 + OW - FR] += 14
a[y0 + FR:y0 + FR + LH, x0 + FR:x0 + FR + LW] = np.asarray(logo, float)
a = np.clip(a, 0, 255)

# 0..31 sütunları duvarın 128..159 kısmında da görünür: oraya hiç taşmamalı
assert np.abs(a[:, :32] - np.clip(tex[:, :32], 0, 255)).max() < 1e-6
assert np.abs(a[176:] - np.clip(tex[176:], 0, 255)).max() < 1e-6


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


blob, q = to_miptex("hp_logo", a)
open(f"{outdir}/hp_logo.miptex", "wb").write(blob)
q.convert("RGB").save(f"{outdir}/hp_logo_256renk.png")
Image.fromarray(a.astype(np.uint8)).crop((0, 0, TW, 176)).save(f"{outdir}/hp_logo_duvar.png")

# ---------------------------------------------------------------- öbekleri yeniden kur
body = tl[4 + 4 * ntex:]
new_offs = [o + 4 if o >= 0 else o for o in toffs]
pad = (-len(tl)) % 4
new_offs.append(len(tl) + 4 + pad)
lumps[2] = bytearray(struct.pack("<i", ntex + 1) + struct.pack("<%di" % (ntex + 1), *new_offs) + body + b"\0" * pad + blob)

new_tix = len(ti) // 40
lumps[6] = ti + new_vecs.astype("<f4").tobytes() + struct.pack("<ii", ntex, flags)
struct.pack_into("<H", fl, FACE * 20 + 10, new_tix)
lumps[7] = fl

order = sorted(range(15), key=lambda i: L[i][0])
out = bytearray(4 + 8 * 15)
struct.pack_into("<i", out, 0, 30)
for i in order:
    while len(out) % 4:
        out += b"\0"
    struct.pack_into("<ii", out, 4 + 8 * i, len(out), len(lumps[i]))
    out += lumps[i]
open(dst, "wb").write(out)
print("yazıldı", dst, len(out), "bayt; yeni doku", ntex, "texinfo", new_tix)
