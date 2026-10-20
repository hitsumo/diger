"""de_tutun.map üretici (Valve 220 biçimi) - Counter-Strike 1.6.

Terk edilmiş tütün fabrikası, gece. Eksenler: +x doğu, +y kuzey, zemin z=0.
  Güney  : T girişi (ana kapı, yükleme yolu)
  Batı   : fabrika ana binası (çatısı çökmüş, asma kat, B bölgesi)
  Orta   : avlu, ortada tırmanılabilir büyük makine
  Doğu   : iki depo (A: barut deposu), aralarında ara sokak, doğu sokağı
  Kuzey  : CT yükleme alanı (batı) ve çöp sahası (doğu); bacanın çevresinde
           sarmal merdiven -> yürüyüş yolu -> depo çatıları -> çatı deliklerinden içeri
"""
import itertools
import math
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WAD_PATH = os.path.join(HERE, "tutun.wad")


def wad_sizes(path):
    d = open(path, "rb").read()
    magic, n, off = struct.unpack_from("<4sii", d, 0)
    assert magic == b"WAD3", path
    sizes = {}
    for i in range(n):
        fo, _, _, _, _, _, name = struct.unpack_from("<iiiBBH16s", d, off + 32 * i)
        sizes[name.split(b"\0")[0].decode()] = struct.unpack_from("<II", d, fo + 16)
    return sizes


TEXSIZE = wad_sizes(WAD_PATH)
TEXSIZE.update({"sky": (16, 16), "aaatrigger": (16, 16), "clip": (16, 16), "null": (16, 16)})
WADS = "tutun.wad;sdhlt.wad"     # derleme: sdHLCSG -wadinclude tutun


# ---------------------------------------------------------------- brushes
class Brush:
    def __init__(self, verts, tex, align="world", scale=(1.0, 1.0), fit=None):
        self.verts = [tuple(int(round(c)) for c in v) for v in verts]
        self.tex = tex
        self.align = align
        self.scale = scale
        self.fit = fit

    def faces(self):
        P = np.array(self.verts, float)
        planes = []
        n_v = len(P)
        for i, j, k in itertools.combinations(range(n_v), 3):
            n = np.cross(P[j] - P[i], P[k] - P[i])
            ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n /= ln
            d = n @ P[i]
            s = P @ n - d
            if np.all(s <= 1e-4):
                pass
            elif np.all(s >= -1e-4):
                n, d = -n, -d
            else:
                continue
            if any(np.allclose(n, q[0], atol=1e-6) and abs(d - q[1]) < 1e-3 for q in planes):
                continue
            on = [m for m in range(n_v) if abs(P[m] @ n - d) < 1e-3]
            planes.append((n, d, (i, j, k), on))
        assert len(planes) >= 4, self.verts
        out = []
        for n, d, (i, j, k), on in planes:
            a, b, cc = P[i], P[j], P[k]
            if np.cross(a - b, cc - b) @ n < 0:
                a, cc = cc, a
            out.append((n, (a, b, cc), P[on]))
        return out

    def tex_for(self, n):
        t = self.tex
        if isinstance(t, str):
            return t
        if n[2] > 0.7:
            key = "top"
        elif n[2] < -0.7:
            key = "bottom"
        else:
            ax = int(np.argmax(np.abs(n[:2])))
            key = ("+" if n[ax] > 0 else "-") + "xy"[ax]
            if key not in t and n[2] > 0.2 and "slope" in t:
                key = "slope"
        if key in t:
            return t[key]
        if key in ("top", "bottom") and "tb" in t:
            return t["tb"]
        return t.get("side", t.get("all", "tf_cwall"))

    def to_map(self):
        lines = ["{"]
        for n, (a, b, c), fv in self.faces():
            tex = self.tex_for(n)
            if abs(n[2]) > 0.99:
                u = np.array([1.0, 0, 0])
                v = np.array([0, -1.0, 0])
            else:
                u = np.array([-n[1], n[0], 0.0])
                u /= np.linalg.norm(u)
                v = np.cross(n, u)
                if v[2] > 0:
                    v = -v
                v /= np.linalg.norm(v)
            tw, th = TEXSIZE.get(tex, (64, 64))
            su, sv = self.scale
            us, vs = fv @ u, fv @ v
            fit = self.fit
            if fit and tex not in ("sky", "aaatrigger", "clip", "null"):
                if fit in ("v", "uv"):
                    sv = (vs.max() - vs.min()) / th
                    su = sv if fit == "v" else su
                if fit in ("u", "uv"):
                    su = (us.max() - us.min()) / tw
            if self.align == "local" or fit:
                ou = (-us.min() / su) % tw
                ov = (-vs.min() / sv) % th
            else:
                ou = ov = 0
            fmt = lambda p: "( %g %g %g )" % tuple(round(x, 3) for x in p)
            lines.append(
                f"{fmt(a)} {fmt(b)} {fmt(c)} {tex} "
                f"[ {u[0]:g} {u[1]:g} {u[2]:g} {ou:.3f} ] [ {v[0]:g} {v[1]:g} {v[2]:g} {ov:.3f} ] 0 {su:.4f} {sv:.4f}"
            )
        lines.append("}")
        return "\n".join(lines)


def box(x0, y0, z0, x1, y1, z1, tex, **kw):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return Brush([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], tex, **kw)


def prism(poly, z0, z1, tex, **kw):
    return Brush([(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly], tex, **kw)


def ngon(cx, cy, r, n, rot=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n + rot), cy + r * math.sin(2 * math.pi * i / n + rot))
            for i in range(n)]


def arc_poly(cx, cy, r0, r1, a0, a1):
    """Halka dilimi (derece): iç ve dış yarıçapta 4 köşe."""
    a0, a1 = math.radians(a0), math.radians(a1)
    return [(cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)), (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
            (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)), (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1))]


def pyramid(poly, z0, apex, tex, **kw):
    return Brush([(x, y, z0) for x, y in poly] + [apex], tex, **kw)


def wedge_x(x_lo, x_hi, y0, y1, z0, z_hi, tex, **kw):
    """x_lo'da z0, x_hi'de z_hi yüksekliğine çıkan rampa (dolu kama)."""
    return Brush([(x_lo, y0, z0), (x_lo, y1, z0), (x_hi, y0, z0), (x_hi, y1, z0),
                  (x_hi, y0, z_hi), (x_hi, y1, z_hi)], tex, **kw)


def wedge_y(y_lo, y_hi, x0, x1, z0, z_hi, tex, **kw):
    return Brush([(x0, y_lo, z0), (x1, y_lo, z0), (x0, y_hi, z0), (x1, y_hi, z0),
                  (x0, y_hi, z_hi), (x1, y_hi, z_hi)], tex, **kw)


def wall_x(x0, x1, y0, y1, z0, z1, openings, tex, **kw):
    """y boyunca uzanan duvar (kalınlık x). openings: (oy0, oy1, oz0, oz1) boşlukları."""
    out, cur = [], y0
    for oy0, oy1, oz0, oz1 in sorted(openings):
        if oy0 > cur:
            out.append(box(x0, cur, z0, x1, oy0, z1, tex, **kw))
        if oz0 > z0:
            out.append(box(x0, oy0, z0, x1, oy1, oz0, tex, **kw))
        if oz1 < z1:
            out.append(box(x0, oy0, oz1, x1, oy1, z1, tex, **kw))
        cur = oy1
    if cur < y1:
        out.append(box(x0, cur, z0, x1, y1, z1, tex, **kw))
    return out


def wall_y(y0, y1, x0, x1, z0, z1, openings, tex, **kw):
    out, cur = [], x0
    for ox0, ox1, oz0, oz1 in sorted(openings):
        if ox0 > cur:
            out.append(box(cur, y0, z0, ox0, y1, z1, tex, **kw))
        if oz0 > z0:
            out.append(box(ox0, y0, z0, ox1, y1, oz0, tex, **kw))
        if oz1 < z1:
            out.append(box(ox0, y0, oz1, ox1, y1, z1, tex, **kw))
        cur = ox1
    if cur < x1:
        out.append(box(cur, y0, z0, x1, y1, z1, tex, **kw))
    return out


def bounds(b):
    P = np.array(b.verts)
    return P.min(axis=0), P.max(axis=0)


def carve(brushes, hole):
    """Eksenlere paralel kutu `hole` = (x0, y0, z0, x1, y1, z1) kutu listesinden çıkarılır."""
    hx0, hy0, hz0, hx1, hy1, hz1 = hole
    out = []
    for b in brushes:
        (x0, y0, z0), (x1, y1, z1) = bounds(b)
        if hx0 >= x1 or hx1 <= x0 or hy0 >= y1 or hy1 <= y0 or hz0 >= z1 or hz1 <= z0:
            out.append(b)
            continue
        mk = lambda *c: box(*c, b.tex, align=b.align, scale=b.scale, fit=b.fit)
        cx0, cx1 = max(x0, hx0), min(x1, hx1)
        cy0, cy1 = max(y0, hy0), min(y1, hy1)
        cz0, cz1 = max(z0, hz0), min(z1, hz1)
        if x0 < cx0:
            out.append(mk(x0, y0, z0, cx0, y1, z1))
        if cx1 < x1:
            out.append(mk(cx1, y0, z0, x1, y1, z1))
        if y0 < cy0:
            out.append(mk(cx0, y0, z0, cx1, cy0, z1))
        if cy1 < y1:
            out.append(mk(cx0, cy1, z0, cx1, y1, z1))
        if z0 < cz0:
            out.append(mk(cx0, cy0, z0, cx1, cy1, cz0))
        if cz1 < z1:
            out.append(mk(cx0, cy0, cz1, cx1, cy1, z1))
    return out


# ---------------------------------------------------------------- entities
world = []
ents = []


def ent(classname, origin=None, brushes=None, **props):
    e = {"classname": classname}
    if origin is not None:
        e["origin"] = "%d %d %d" % tuple(int(round(c)) for c in origin)
    e.update({k: str(v) for k, v in props.items()})
    ents.append((e, brushes or []))


def W(*bs):
    for b in bs:
        if isinstance(b, list):
            world.extend(b)
        else:
            world.append(b)


def light(origin, rgb, brightness, style=0, **kw):
    props = {"_light": "%d %d %d %d" % (*rgb, brightness)}
    if style:
        props["style"] = style
    props.update(kw)
    ent("light", origin, **props)


def sound(origin, wav, volume=7, flags=4, **kw):
    """ambient_generic: flags 1=her yerde 2=küçük 4=orta 8=büyük yarıçap 16=sessiz başla 32=tekrarsız."""
    ent("ambient_generic", origin, message=wav, health=volume, spawnflags=flags, pitch=100, **kw)


def breakable(brush, material, health, **kw):
    ent("func_breakable", brushes=[brush], material=material, health=health, **kw)


def fence_wall(brushes):
    ent("func_wall", brushes=brushes, rendermode=4, renderamt=255)


def crate(x, y, z, w=64, d=64, h=48, breakable_=True):
    b = box(x, y, z, x + w, y + d, z + h, "tf_crate", align="local")
    if breakable_:
        breakable(b, 1, 40)
    else:
        W(b)


def barrel(cx, cy, z=0, h=56, explosive=False):
    b = prism(ngon(cx, cy, 20, 8, math.pi / 8), z, z + h, {"side": "tf_barrel", "tb": "tf_barreltop"}, fit="v")
    if explosive:
        breakable(b, 2, 60, explodemagnitude=110)
    else:
        W(b)


def sacks(x0, y0, x1, y1, h=40):
    W(box(x0, y0, 0, x1, y1, h, "tf_sack", align="local"))


def lamp_post(x, y, direction, rgb=(255, 176, 86), brightness=400, h=200):
    dx, dy = direction
    W(box(x - 5, y - 5, 0, x + 5, y + 5, h, "tf_steel"),
      box(min(x, x + 48 * dx) - 4, min(y, y + 48 * dy) - 4, h - 8, max(x, x + 48 * dx) + 4, max(y, y + 48 * dy) + 4, h, "tf_steel"),
      box(x + 48 * dx - 14, y + 48 * dy - 14, h - 16, x + 48 * dx + 14, y + 48 * dy + 14, h - 8,
          {"bottom": "tf_lamp", "side": "tf_steel", "top": "tf_steel"}))
    light((x + 48 * dx, y + 48 * dy, h - 28), rgb, brightness)
    sound((x + 48 * dx, y + 48 * dy, h - 30), "tutun/vizilti.wav", 4, 2)


def graffiti(face, tex, a0, a1, z0, z1, c):
    """Duvar yüzeyine 4 birim kabartma grafiti. face: '+x' (yüzey normali +x, duvar c'de biter) vb."""
    if face == "+x":
        W(box(c, a0, z0, c + 4, a1, z1, tex, align="local", fit="uv"))
    elif face == "-x":
        W(box(c - 4, a0, z0, c, a1, z1, tex, align="local", fit="uv"))
    elif face == "+y":
        W(box(a0, c, z0, a1, c + 4, z1, tex, align="local", fit="uv"))
    else:
        W(box(a0, c - 4, z0, a1, c, z1, tex, align="local", fit="uv"))


# ================================================================ DÜNYA
X0, X1, Y0, Y1, Z0, Z1 = -1792, 1792, -1792, 1792, -128, 1024
T = 32
W(box(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T, "sky"),
  box(X0 - T, Y0 - T, Z0 - T, X1 + T, Y1 + T, Z0, "tf_dirt"),
  box(X0 - T, Y0, Z0, X0, Y1, Z1, "sky"),
  box(X1, Y0, Z0, X1 + T, Y1, Z1, "sky"),
  box(X0 - T, Y0 - T, Z0, X1 + T, Y0, Z1, "sky"),
  box(X0 - T, Y1, Z0, X1 + T, Y1 + T, Z1, "sky"))

# ---- zemin: taban + bölge kaplamaları (üst üste binmeyen döşeme)
GROUND_TEX = {"gravel": {"top": "tf_gravel", "side": "tf_dirt"}, "asphalt": {"top": "tf_asphalt", "side": "tf_dirt"},
              "concrete": {"top": "tf_concrete", "side": "tf_cwall"}, "dirt": {"top": "tf_dirt", "side": "tf_dirt"}}
W(box(X0, Y0, Z0, X1, Y1, -8, "tf_dirt"))
REGIONS = [
    (X0, Y0, -1536, Y1, "dirt"),
    (-1536, Y0, -640, -704, "gravel"), (-1536, -704, -640, 576, "concrete"), (-1536, 576, -640, Y1, "concrete"),
    (-640, Y0, 704, -704, "gravel"), (-640, -704, 704, 704, "asphalt"), (-640, 704, 704, 832, "gravel"),
    (-640, 832, 256, Y1, "concrete"), (256, 832, 704, Y1, "dirt"),
    (704, Y0, 1344, -672, "gravel"), (704, -672, 1344, -192, "concrete"), (704, -192, 1344, 192, "asphalt"),
    (704, 192, 1344, 672, "concrete"), (704, 672, 1344, 832, "gravel"), (704, 832, 1344, Y1, "dirt"),
    (1344, Y0, X1, 832, "gravel"), (1344, 832, X1, Y1, "dirt"),
]
GROUND = [box(x0, y0, -8, x1, y1, 0, GROUND_TEX[t]) for x0, y0, x1, y1, t in REGIONS]

# ---- çevre duvarı (tuğla, 160 yüksek) ve güney ana kapı
PW = {"side": "tf_brick2", "top": "tf_trim", "bottom": "tf_brick2"}
W(wall_y(-1568, -1536, -1568, 1568, 0, 160, [(-160, 160, 0, 160)], PW),
  box(-1568, 1536, 0, 1568, 1568, 160, PW),
  box(-1568, -1536, 0, -1536, 1536, 160, PW),
  box(1536, -1536, 0, 1568, 1536, 160, PW))
# kapı: kapalı demir kapı (tel örgü dokulu), üstünde tabela ve çerçeve
fence_wall([box(-160, -1556, 0, 160, -1552, 160, "{tf_fence", align="local")])
W(box(-176, -1572, 0, -160, -1532, 240, "tf_steel"), box(160, -1572, 0, 176, -1532, 240, "tf_steel"),
  box(-176, -1572, 232, 176, -1532, 240, "tf_steel"),
  box(-128, -1556, 168, 128, -1550, 232, "tf_sign", align="local", fit="uv"))
graffiti("+y", "tf_graf1", 300, 556, 16, 144, -1536)
graffiti("+y", "tf_graf3", -900, -644, 16, 144, -1536)
graffiti("-x", "tf_graf2", -1100, -844, 16, 144, 1536)

# ================================================================ FABRİKA (batı, B)
FX0, FX1, FY0, FY1, FZ = -1536, -640, -704, 576, 352
EXT = {"side": "tf_brick", "tb": "tf_brick"}
INT = {"side": "tf_plaster", "tb": "tf_plaster"}
# doğu duvarı (avluya bakar): büyük kapı + kırık pencereler
E_WIN = [(-544, -416), (-352, -224), (224, 352), (416, 544)]
W(wall_x(-672, -640, FY0, FY1, 0, FZ, [(-96, 96, 0, 192)] + [(a, b, 96, 224) for a, b in E_WIN],
         {"+x": "tf_brick", "-x": "tf_plaster", "side": "tf_brick", "tb": "tf_brick"}))
for a, b in E_WIN:                       # üst kat pencereleri (doku) + alt pencere eşikleri
    W(box(-640, a, 240, -636, b, 336, "tf_window", align="local", fit="uv"),
      box(-676, a - 8, 92, -636, b + 8, 96, "tf_trim"))
W(box(-640, -128, 200, -636, 128, 264, "tf_sign", align="local", fit="uv"))      # kapı üstü tabela
# güney duvarı (T yoluna bakar): kapı + bir pencere ; kuzey duvarı (CT alanına): kapı
W(wall_y(-704, -672, FX0, FX1, 0, FZ, [(-1000, -872, 0, 160), (-1400, -1272, 96, 224)],
         {"-y": "tf_brick", "+y": "tf_plaster", "side": "tf_brick", "tb": "tf_brick"}))
W(wall_y(544, 576, FX0, FX1, 0, FZ, [(-1000, -872, 0, 160), (-800, -672, 96, 224)],
         {"+y": "tf_brick", "-y": "tf_plaster", "side": "tf_brick", "tb": "tf_brick"}))
W(box(-1536, FY0, 0, -1504, FY1, FZ, {"-x": "tf_brick2", "+x": "tf_plaster", "side": "tf_brick", "tb": "tf_brick"}))
for a, b in ((-1400, -1272), (-1200, -1072), (-800, -672)):
    W(box(a, -708, 240, b, -704, 336, "tf_window", align="local", fit="uv"),
      box(a, 576, 240, b, 580, 336, "tf_window", align="local", fit="uv"))
# çatı: plaka + çökmüş iki delik, korkuluk duvarı
ROOF = [box(FX0, FY0, FZ, FX1, FY1, FZ + 16, {"top": "tf_roof", "bottom": "tf_roof", "side": "tf_steel"})]
for hole in ((-1300, -560, FZ - 1, -860, -180, FZ + 17), (-1200, 200, FZ - 1, -1000, 400, FZ + 17)):
    ROOF = carve(ROOF, hole)
W(ROOF)
W(box(FX0, FY0, FZ + 16, FX1, FY0 + 32, FZ + 48, EXT), box(FX0, FY1 - 32, FZ + 16, FX1, FY1, FZ + 48, EXT),
  box(FX0, FY0, FZ + 16, FX0 + 32, FY1, FZ + 48, EXT), box(FX1 - 32, FY0, FZ + 16, FX1, FY1, FZ + 48, EXT))
# kolonlar
for cx in (-1216, -960):
    for cy in (-512, -256, 0, 256):
        W(box(cx - 16, cy - 16, 0, cx + 16, cy + 16, FZ, "tf_steel"))
# asma kat (batı duvarı boyunca), korkuluk, merdiven (kuzey duvarı boyunca), çökmüş döşeme rampası (güney)
MZ = 176
W(box(-1504, -672, MZ - 16, -1344, 544, MZ, {"top": "tf_catwalk", "bottom": "tf_steel", "side": "tf_steel"}),
  box(-1344, -672, MZ, -1336, -640, MZ + 40, "tf_steel"),
  box(-1344, -560, MZ, -1336, 480, MZ + 40, "tf_steel"))
for k in range(1, 12):
    W(box(-896 - 32 * k, 480, 0, -896 - 32 * (k - 1), 544, 16 * k, {"top": "tf_concrete", "side": "tf_cwall"}))
W(box(-1344, 480, MZ - 16, -1248, 544, MZ, {"top": "tf_catwalk", "bottom": "tf_steel", "side": "tf_steel"}))
W(box(-896, 472, 0, -1248, 480, 56, "tf_steel"))                   # merdiven korkuluğu (alçak)
W(wedge_x(-896, -1344, -640, -560, 0, MZ, {"top": "tf_rubble", "slope": "tf_rubble", "side": "tf_cwall"}))
# moloz yığınları (çatı deliklerinin altı), beton parçaları
W(pyramid(ngon(-1080, -370, 130, 8), 0, (-1080, -370, 60), "tf_rubble"),
  pyramid(ngon(-940, -240, 80, 8), 0, (-940, -240, 40), "tf_rubble"),
  pyramid(ngon(-1100, 300, 70, 8), 0, (-1100, 300, 36), "tf_rubble"),
  box(-1250, -480, 0, -1170, -416, 40, "tf_cwall"), box(-1000, -470, 0, -940, -430, 24, "tf_cwall"),
  box(-760, 40, 0, -696, 104, 48, "tf_cwall"))
# fabrika içi eşyalar: sandıklar, fıçılar, çuvallar, kırık makine
crate(-1480, -100, 0); crate(-1480, -36, 0); crate(-1480, -100, 48)
crate(-800, -600, 0, 64, 64, 48); crate(-736, -600, 0)
barrel(-700, 300); barrel(-700, 340); barrel(-740, 320)
sacks(-1200, 480, -1040, 540, 36)
W(box(-1300, 100, 0, -1100, 220, 96, "tf_panel"), box(-1280, 220, 0, -1120, 260, 64, "tf_metal"))   # eski kurutma makinesi
# grafitiler (iç ve dış)
graffiti("+x", "tf_graf3", -400, -144, 24, 152, -1504)
graffiti("+x", "tf_graf1", 100, 356, 24, 152, -1504)
graffiti("-y", "tf_graf2", -1300, -1044, 24, 152, 544)
graffiti("+y", "tf_graf4", -1260, -1004, 24, 152, -672)
graffiti("+y", "tf_graf2", -1400, -1144, 16, 144, 576)
# avlu tarafı pencere: dışarıda ve içeride moloz rampası (tırmanılabilir kırık pencere)
W(wedge_x(-560, -640, -340, -236, 0, 96, {"top": "tf_rubble", "slope": "tf_rubble", "side": "tf_rubble"}),
  wedge_x(-760, -672, -340, -236, 0, 96, {"top": "tf_rubble", "slope": "tf_rubble", "side": "tf_rubble"}))
W(box(-644, -660, 24, -640, -596, 88, "tf_signb", align="local", fit="uv"))

# ================================================================ AVLU ve MAKİNE (orta)
# makine: kaide -> basamaklı metal bloklar -> gövde -> üst kat ; kazan, volan, borular
W(box(-256, -192, 0, 256, 192, 32, {"top": "tf_concrete", "side": "tf_cwall"}),
  box(-288, -96, 0, -256, 96, 16, {"top": "tf_concrete", "side": "tf_cwall"}),
  box(256, -96, 0, 288, 96, 16, {"top": "tf_concrete", "side": "tf_cwall"}),
  box(96, -192, 32, 256, -96, 64, "tf_metal"),                       # blok A (64)
  box(160, -96, 32, 256, 32, 96, "tf_metal"),                        # blok B (96)
  box(-192, -128, 32, 160, 128, 128, "tf_panel"),                    # gövde (128)
  box(-160, -128, 128, 0, 0, 160, "tf_panel"),                       # üst kat (160)
  box(-192, 40, 128, 160, 120, 136, "tf_steel"))                     # kazan yatağı
# kazan: x ekseninde yatan sekizgen silindir
cy, cz, r = 64, 184, 56
ring = [(cy + r * math.cos(a), cz + r * math.sin(a)) for a in [2 * math.pi * i / 8 + math.pi / 8 for i in range(8)]]
W(Brush([(-192, y, z) for y, z in ring] + [(160, y, z) for y, z in ring], "tf_metal"))
W(prism(ngon(-100, 64, 14, 8), 232, 420, "tf_pipe"))                 # kazan bacası
# volan (dikey çark, y ekseni)
wheel = [(-224 + 96 * math.cos(a), 96 + 96 * math.sin(a)) for a in [2 * math.pi * i / 16 for i in range(16)]]
W(Brush([(x, -12, z) for x, z in wheel] + [(x, 12, z) for x, z in wheel], "tf_steel"))
W(box(-236, -24, 84, -212, 24, 108, "tf_steel"))
# borular: gövdeden yukarı ve batıya (baş üstü)
W(box(96, 96, 128, 112, 112, 320, "tf_pipe"), box(-520, 96, 304, 112, 112, 320, "tf_pipe"),
  box(-520, 96, 0, -504, 112, 320, "tf_pipe"), box(-520, -520, 304, -504, 112, 320, "tf_pipe"))
# avlu lambaları (sodyum, turuncu) ve eşyalar
lamp_post(-600, 640, (1, -0.3))
lamp_post(640, -640, (-1, 0.3))
crate(360, 400, 0); crate(424, 400, 0); crate(392, 400, 48)
barrel(-440, -520); barrel(-480, -560, explosive=True)
W(box(-560, -200, 0, -480, -40, 56, "tf_metal"))                     # devrilmiş dolap
sacks(500, -120, 580, 0, 32)

# ================================================================ DEPOLAR (doğu)
DW = {"side": "tf_cwall", "tb": "tf_cwall"}
ROOF_T = {"top": "tf_roof", "bottom": "tf_steel", "side": "tf_steel"}


def depot(x0, y0, x1, y1, front_door, front_window, south_open, north_open, east_open, roof_hole, lip_gaps, tilt):
    """Depo kabuğu. front_door=(ya, yb) batı duvarında, gaps: (ya, yb) çatı kenar bordüründe boşluklar."""
    h = 224
    W(wall_x(x0, x0 + 24, y0, y1, 0, h, [(front_door[0], front_door[1], 0, 192)] + [(a, b, 64, 112) for a, b in front_window], DW))
    W(wall_y(y0, y0 + 24, x0 + 24, x1 - 24, 0, h, south_open, DW))
    W(wall_y(y1 - 24, y1, x0 + 24, x1 - 24, 0, h, north_open, DW))
    W(wall_x(x1 - 24, x1, y0, y1, 0, h, east_open, DW))
    roof = [box(x0, y0, h, x1, y1, h + 16, ROOF_T)]
    roof = carve(roof, (roof_hole[0], roof_hole[1], h - 1, roof_hole[2], roof_hole[3], h + 17))
    W(roof)
    # çatı bordürü (16 yüksek), yürüyüş yolu / tahta köprü girişlerinde boşluk
    lip = [box(x0, y0, h + 16, x1, y0 + 8, h + 32, "tf_cwall"), box(x0, y1 - 8, h + 16, x1, y1, h + 32, "tf_cwall"),
           box(x0, y0, h + 16, x0 + 8, y1, h + 32, "tf_cwall"), box(x1 - 8, y0, h + 16, x1, y1, h + 32, "tf_cwall")]
    for gx0, gy0, gx1, gy1 in lip_gaps:
        lip = carve(lip, (gx0, gy0, h + 15, gx1, gy1, h + 33))
    W(lip)
    # sıkışmış sürgülü kapı: açıklığın büyük kısmını kapatan, hafif yan yatmış panel
    ya, yb = front_door
    if tilt > 0:
        pa, pb = ya, yb - 56
    else:
        pa, pb = ya + 56, yb
    W(Brush([(x0 - 4, pa, 0), (x0 - 4, pb, 0), (x0 + 28, pa, 0), (x0 + 28, pb, 0),
             (x0 - 4 + 20, pa - 10 * tilt, 192), (x0 - 4 + 20, pb - 10 * tilt, 192),
             (x0 + 28 + 20, pa - 10 * tilt, 192), (x0 + 28 + 20, pb - 10 * tilt, 192)],
            "tf_door", align="local", fit="uv"))
    W(box(x0 - 8, ya - 16, 192, x0 + 32, yb + 16, 204, "tf_steel"))      # kapı rayı


# Depo 1 (A, barut): ön kapı y -480..-320 (boşluk -376..-320), arka kapı kuzeyde, doğu kapısı sokağa
depot(704, -672, 1344, -192, (-480, -320), [(-256, -208)],
      [(800, 864, 64, 112), (1184, 1248, 64, 112)], [(1216, 1280, 0, 128)],
      [(-300, -236, 0, 128), (-500, -436, 64, 112)], (960, -528, 1152, -368),
      [(992, -200, 1072, -184), (696, -608, 712, -560)], tilt=1)
W(box(700, -310, 96, 704, -246, 160, "tf_signa", align="local", fit="uv"))
graffiti("-x", "tf_graf2", -664, -488, 120, 184, 704)
graffiti("+y", "tf_graf4", 800, 1056, 16, 144, -192)
# içi: çatı deliği altında sandık kulesi, barut fıçıları (patlar), sandıklar, çuvallar
crate(1000, -480, 0, 96, 64, 48, breakable_=False); crate(1016, -480, 48, 64, 64, 48, breakable_=False)
crate(1240, -660, 0); crate(1240, -596, 0); crate(1240, -660, 48)
crate(760, -420, 0); crate(760, -356, 0)
for bx, by in ((780, -620), (820, -640), (800, -580), (1280, -300), (1250, -260), (1120, -660)):
    barrel(bx, by, explosive=True)
barrel(900, -250); barrel(940, -230)
sacks(1100, -660, 1200, -600, 40); sacks(880, -660, 980, -620, 32)
W(box(740, -640, 0, 760, -560, 120, "tf_steel"), box(740, -640, 120, 900, -560, 128, "tf_steel"))   # raf
# yıkık tahta perde: doğu penceresi (kırılabilir)
breakable(box(1316, -500, 64, 1320, -436, 112, "tf_boards", align="local"), 1, 25)
# Depo 1 çatısından avluya merdiven (batı yüz)
ent("func_illusionary", brushes=[box(700, -600, 0, 704, -568, 240, "{tf_ladder", align="local")], rendermode=4, renderamt=255)
ent("func_ladder", brushes=[box(686, -600, 0, 704, -568, 250, "aaatrigger")])

# Depo 2: ön kapı y 320..480 (boşluk 320..376), arka kapı güneyde (ara sokak), kuzey kapısı yola
depot(704, 192, 1344, 672, (320, 480), [(560, 608)],
      [(1216, 1280, 0, 128)], [(1216, 1280, 0, 128), (800, 864, 64, 112)],
      [(400, 464, 64, 112)], (900, 424, 1052, 576),
      [(992, 184, 1072, 200), (768, 664, 832, 680)], tilt=-1)
graffiti("-x", "tf_graf3", 488, 664, 120, 184, 704)
graffiti("+y", "tf_graf1", 800, 1056, 16, 144, 192)
crate(940, 460, 0, 96, 64, 48, breakable_=False); crate(956, 460, 48, 64, 64, 48, breakable_=False)
sacks(740, 220, 900, 320, 44); sacks(1180, 560, 1300, 640, 36); sacks(760, 560, 860, 640, 40)
crate(1240, 230, 0); crate(1240, 294, 0); crate(1180, 230, 0)
barrel(1280, 500); barrel(1240, 530)
W(box(1000, 600, 0, 1120, 640, 72, "tf_metal"))                      # el arabası/dolap
# tahta köprü: iki depo çatısı arasında
W(box(1000, -192, 232, 1064, 192, 240, "tf_wood", align="local"))
# ara sokak: fıçılar, konteyner
W(box(900, 20, 0, 1060, 120, 96, "tf_metal"))                        # çöp konteyneri
barrel(760, -140); barrel(1280, 140); barrel(1300, 100)
crate(1180, -170, 0)

# ================================================================ ÇÖP SAHASI, BACA, SARMAL MERDİVEN (kuzeydoğu)
CX, CY = 800, 1200
W(prism(ngon(CX, CY, 72, 16), 0, 640, "tf_brick2", align="local"),
  prism(ngon(CX, CY, 84, 16), 640, 656, "tf_trim"),
  prism(ngon(CX, CY, 80, 16), 300, 308, "tf_steel"))                  # çember bant
# bacaya asılı çelik sarmal merdiven: her basamak 16 kalın bir halka dilimi, dış kenarda korkuluk
for k in range(1, 16):
    a0 = 292.5 + 22.5 * (k - 1)
    W(prism(arc_poly(CX, CY, 64, 184, a0, a0 + 22.5), 16 * k - 16, 16 * k, {"top": "tf_catwalk", "side": "tf_steel", "bottom": "tf_steel"}),
      prism(arc_poly(CX, CY, 180, 188, a0, a0 + 22.5), 16 * k, 16 * k + 40, "tf_steel"))
# üst sahanlık (güney) ve yürüyüş yolu -> Depo 2 çatısı
W(Brush([(730, 1030, 224), (800, 1016, 224), (856, 1016, 224), (856, 960, 224), (730, 960, 224),
         (730, 1030, 240), (800, 1016, 240), (856, 1016, 240), (856, 960, 240), (730, 960, 240)],
        {"top": "tf_catwalk", "side": "tf_steel", "bottom": "tf_steel"}))
W(box(776, 672, 224, 824, 960, 240, {"top": "tf_catwalk", "side": "tf_steel", "bottom": "tf_steel"}),
  box(772, 672, 240, 776, 960, 272, "tf_steel"), box(824, 672, 240, 828, 960, 272, "tf_steel"),
  box(730, 960, 240, 734, 1030, 272, "tf_steel"), box(856, 960, 240, 860, 1016, 272, "tf_steel"))
for py in (760, 880):
    W(box(792, py - 8, 0, 808, py + 8, 224, "tf_steel"))
# çöp yığınları, branda, konteyner, lastik/sandık
for cx_, cy_, r_, h_ in ((460, 1000, 150, 64), (520, 1420, 150, 90), (1100, 1380, 150, 100),
                         (1300, 1000, 140, 72), (1050, 940, 80, 40), (380, 1420, 110, 56), (1250, 1250, 110, 60)):
    W(pyramid(ngon(cx_, cy_, r_, 8), 0, (cx_, cy_, h_), "tf_garbage"))
W(box(1360, 1420, 0, 1500, 1520, 56, "tf_tarp", align="local"),
  box(300, 1250, 0, 460, 1378, 112, "tf_metal"),
  box(1380, 880, 0, 1500, 960, 48, "tf_rubble"))
crate(1420, 1180, 0); crate(1420, 1244, 0)
barrel(360, 1150); barrel(400, 1120)
# tel örgüler: CT alanı ile çöp sahası arası (x=256) ve kuzey yolu ile çöp sahası arası (y=832)
fence_wall([box(254, 832, 0, 258, 1100, 128, "{tf_fence", align="local"),
            box(254, 1200, 0, 258, 1536, 128, "{tf_fence", align="local"),
            box(256, 830, 0, 560, 834, 128, "{tf_fence", align="local"),
            box(680, 830, 0, 1180, 834, 128, "{tf_fence", align="local"),
            box(1300, 830, 0, 1344, 834, 128, "{tf_fence", align="local")])
for px, py in ((256, 832), (256, 960), (256, 1100), (256, 1200), (256, 1340), (256, 1480),
               (400, 832), (560, 832), (680, 832), (840, 832), (1000, 832), (1180, 832), (1300, 832)):
    W(box(px - 4, py - 4, 0, px + 4, py + 4, 136, "tf_steel"))
# baca projektörü (titrer)
W(box(CX + 72, CY - 12, 320, CX + 96, CY + 12, 336, {"bottom": "tf_lamp", "side": "tf_steel", "top": "tf_steel"}))
ent("light_spot", (CX + 84, CY, 316), _light="210 215 255 500", pitch=-80, _cone=50, _cone2=70, style=10)

# ================================================================ CT ALANI (kuzeybatı) ve T YOLU (güney)
W(box(-1000, 1300, 0, -800, 1428, 128, "tf_metal"),                  # konteyner
  box(-1500, 1300, 0, -1372, 1428, 96, "tf_metal"))
crate(-700, 1300, 0); crate(-636, 1300, 0); crate(-668, 1300, 48)
sacks(-400, 1400, -280, 1500, 40)
lamp_post(-300, 1000, (-0.3, 1))
# T yolu
W(box(-400, -1300, 0, -272, -1172, 48, "tf_wood", align="local"),   # paletler
  box(-400, -1300, 48, -336, -1236, 96, "tf_wood", align="local"),
  box(900, -1400, 0, 1160, -1300, 80, "tf_metal"), box(1160, -1400, 0, 1240, -1300, 120, "tf_panel"))  # hurda kamyon
crate(-1200, -1200, 0); crate(-1136, -1200, 0)
barrel(1300, -1000); barrel(1340, -1020); barrel(-700, -900, explosive=True)
lamp_post(640, -1480, (-0.3, 1))
lamp_post(-1200, -800, (0.3, -1))

# ================================================================ OYUN VARLIKLARI
# T: güney kapı içi ; CT: kuzeybatı yükleme alanı
n = 0
for y in (-1460, -1400, -1340, -1280):
    for x in (120, 200, 280, 360):
        ent("info_player_deathmatch", (x, y, 40), angles="0 90 0"); n += 1
for y in (1000, 1080, 1160, 1240):
    for x in (-1380, -1300, -1220, -1140):
        ent("info_player_start", (x, y, 40), angles="0 270 0")
ent("func_bomb_target", brushes=[box(740, -660, 0, 1310, -220, 200, "aaatrigger")])         # A: barut deposu
ent("func_bomb_target", brushes=[box(-1500, -680, 0, -900, -220, 220, "aaatrigger")])       # B: fabrika güney salonu
ent("info_map_parameters", (0, 0, 600), bombradius=400)

# ================================================================ IŞIKLAR
ent("light_environment", (0, 0, 900), angles="0 230 0", pitch=-50,
    _light="125 140 190 60", _diffuse_light="50 60 100 30")
# fabrika: sarkıt ampuller (biri titrer) + asma kat altı
light((-1100, -400, 300), (255, 206, 150), 280)
light((-800, -100, 300), (255, 206, 150), 200)
light((-900, 250, 300), (255, 206, 150), 220, style=6)
light((-1420, -200, 150), (255, 206, 150), 160)
light((-1424, 300, 262), (255, 206, 150), 170)                        # asma kat üstü (döşeme 176, çatı 352)
light((-1300, 500, 260), (255, 206, 150), 140, style=6)               # merdiven başı, titrer
# depolar
light((1024, -430, 200), (255, 210, 150), 300, style=10)
light((1250, -250, 150), (255, 210, 150), 180)
light((900, -400, 190), (255, 210, 150), 160)
light((1000, 430, 200), (255, 210, 150), 260)
light((1240, 600, 190), (255, 210, 150), 160)
light((1000, 0, 180), (255, 210, 150), 180, style=6)                  # ara sokak duvar lambası
W(box(992, 184, 170, 1008, 192, 186, {"-y": "tf_lamp", "side": "tf_steel", "tb": "tf_steel"}))
# makine: iş lambası (boru üstünde) ve kırmızı uyarı lambası
light((96, 96, 290), (255, 220, 160), 220)
light((-60, -60, 200), (255, 60, 40), 80, style=11)
# baca: sarmal merdiven boyunca iki lamba ve alt uç
for ang, z in ((22.5, 140), (157.5, 236)):
    light((CX + 96 * math.cos(math.radians(ang)), CY + 96 * math.sin(math.radians(ang)), z), (255, 200, 130), 180)
light((940, 1060, 60), (255, 190, 110), 200)

# ================================================================ SESLER
for pos in ((0, 0, 260), (-300, -1200, 200), (-700, 1200, 200), (1440, 0, 200), (900, 1300, 300)):
    sound(pos, "tutun/ruzgar.wav", 6, 8)
for pos in ((-1100, -380, 120), (-1000, 300, 120), (1100, -300, 100), (1000, 560, 100)):
    sound(pos, "tutun/damla.wav", 7, 2)
for pos in ((800, 1000, 240), (1000, 0, 240), (-1420, 0, 200)):
    sound(pos, "tutun/gicirti.wav", 6, 4)
for pos in ((460, 1000, 60), (1100, 1380, 80), (1250, 1250, 60)):
    sound(pos, "tutun/sinek.wav", 6, 2)
sound((0, 0, 120), "tutun/ugultu.wav", 7, 4)
# aralıklı tek sesler: kendini tetikleyen multi_manager (çok kanallı bayrak) döngüsü
sound((0, 0, 500), "tutun/gumburtu.wav", 8, 1 + 16 + 32, targetname="ses_gok")
sound((900, 1200, 100), "tutun/carpma.wav", 8, 4 + 16 + 32, targetname="ses_carpma1")
sound((-1000, -100, 150), "tutun/carpma.wav", 7, 4 + 16 + 32, targetname="ses_carpma2")
sound((-700, 100, 200), "tutun/cam.wav", 7, 4 + 16 + 32, targetname="ses_cam")
sound((1200, -400, 100), "tutun/cam.wav", 6, 4 + 16 + 32, targetname="ses_cam2")
ents.append(({"classname": "multi_manager", "targetname": "mm_sesler", "spawnflags": "1",
              "ses_gok": "9", "ses_carpma1": "21", "ses_cam": "33", "ses_carpma2": "44",
              "ses_cam2": "52", "mm_sesler": "61"}, []))
ent("trigger_auto", (0, 0, 64), target="mm_sesler", spawnflags=1)

# ================================================================ YAZ
W(GROUND)


def write(path):
    out = ['{', '"classname" "worldspawn"', '"mapversion" "220"', f'"wad" "{WADS}"',
           '"skyname" "tutun"', '"MaxRange" "6144"', '"message" "Tutun Fabrikasi"', '"sounds" "1"']
    out += [b.to_map() for b in world]
    out.append("}")
    for e, bs in ents:
        out.append("{")
        out += [f'"{k}" "{v}"' for k, v in e.items()]
        out += [b.to_map() for b in bs]
        out.append("}")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"{len(world)} world brushes, {len(ents)} entities")


if __name__ == "__main__":
    write(os.path.join(HERE, "de_tutun.map"))
