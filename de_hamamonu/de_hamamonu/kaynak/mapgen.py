"""de_hamamonu.map üretici (Valve 220) - Counter-Strike 1.6.

Ankara Hamamönü: restore edilmiş Osmanlı mahallesi, gündüz. Eksenler: +x doğu, +y kuzey.
İki kot: aşağı mahalle z=0 (T, Saat Kulesi Meydanı/B, güney sokak, avlu, ara sokaklar),
yukarı mahalle z=128 (ana sokak, kuzey sokak, Hacettepe Önü/A, CT). Aralarda merdivenler
ve içinden geçilen evlerin iç merdivenleri.
"""
import itertools
import math
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WAD_PATH = os.path.join(HERE, "hamamonu.wad")


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
WADS = "hamamonu.wad;sdhlt.wad"     # derleme: sdHLCSG -wadinclude hamamonu


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
        return t.get("side", t.get("all", "ho_pl_w"))

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


# ---------------------------------------------------------------- detay varlıkları (func_wall)
# Küçük süs fırçaları (pencere, kapı, tabela, cumba, bank, lamba, kubbe dilimleri...) dünyayı
# bölmesin diye func_wall gruplarına toplanır: VIS çok hızlanır, oyun içi r_speeds düşer.
DETAIL = []


def group(name, shadow=False):
    DETAIL.append([name, [], shadow])


def D(*bs):
    if not DETAIL:
        group("genel")
    for b in bs:
        if isinstance(b, list):
            DETAIL[-1][1].extend(b)
        elif b is not None:
            DETAIL[-1][1].append(b)


def light(origin, rgb, brightness, style=0, **kw):
    props = {"_light": "%d %d %d %d" % (*rgb, brightness)}
    if style:
        props["style"] = style
    props.update(kw)
    ent("light", origin, **props)


def sound(origin, wav, volume=7, flags=4, **kw):
    ent("ambient_generic", origin, message=wav, health=volume, spawnflags=flags, pitch=100, **kw)


def hip_roof(x0, y0, x1, y1, z0, z1, tex, **kw):
    dx, dy = x1 - x0, y1 - y0
    if dy >= dx:
        m = dx / 2
        ridge = [((x0 + x1) / 2, y0 + m, z1), ((x0 + x1) / 2, y1 - m, z1)]
    else:
        m = dy / 2
        ridge = [(x0 + m, (y0 + y1) / 2, z1), (x1 - m, (y0 + y1) / 2, z1)]
    base = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    if abs(ridge[0][0] - ridge[1][0]) < 1 and abs(ridge[0][1] - ridge[1][1]) < 1:
        ridge = ridge[:1]
    return Brush(base + ridge, tex, **kw)


def relief(face, c, a0, a1, z0, z1, tex, depth=4):
    """Duvar yüzeyine kabartma (pencere, kapı, tabela). face = yüzey normali yönü."""
    if face == "+x":
        return box(c, a0, z0, c + depth, a1, z1, tex, align="local", fit="uv")
    if face == "-x":
        return box(c - depth, a0, z0, c, a1, z1, tex, align="local", fit="uv")
    if face == "+y":
        return box(a0, c, z0, a1, c + depth, z1, tex, align="local", fit="uv")
    return box(a0, c - depth, z0, a1, c, z1, tex, align="local", fit="uv")


COLORS = ["ho_pl_w", "ho_pl_y", "ho_pl_p", "ho_pl_b", "ho_pl_g", "ho_pl_w"]
_color_i = [0]


def next_color():
    c = COLORS[_color_i[0] % len(COLORS)]
    _color_i[0] += 1
    return c


def face_span(face, x0, y0, x1, y1):
    """Yüzün koordinatı (c) ve yüz boyunca aralık (a0, a1)."""
    if face == "-y":
        return y0, x0, x1
    if face == "+y":
        return y1, x0, x1
    if face == "-x":
        return x0, y0, y1
    return x1, y0, y1


def windows_on(face, x0, y0, x1, y1, z0, color_stone=False, count=None, w=48, h=80, tex="ho_window"):
    c, a0, a1 = face_span(face, x0, y0, x1, y1)
    L = a1 - a0
    n = count or max(1, int(L // 112))
    step = L / (n + 1)
    out = []
    for i in range(n):
        m = a0 + step * (i + 1)
        out.append(relief(face, c, m - w / 2, m + w / 2, z0, z0 + h, tex))
    return out


def cumba(face, x0, y0, x1, y1, zf, center, width=128, hollow=False, color="ho_pl_w"):
    """Cumba: üst katta 48 birim dışa taşan kafes pencereli çıkma. zf = üst kat döşeme üstü.
    hollow=True ise içi boş: zemin, tavan, üç yanı pencere delikli duvar (içeriden ateş edilir)."""
    c, a0, a1 = face_span(face, x0, y0, x1, y1)
    lo, hi = center - width / 2, center + width / 2
    d = 48
    if face == "-y":
        bx0, by0, bx1, by1 = lo, c - d, hi, c
    elif face == "+y":
        bx0, by0, bx1, by1 = lo, c, hi, c + d
    elif face == "-x":
        bx0, by0, bx1, by1 = c - d, lo, c, hi
    else:
        bx0, by0, bx1, by1 = c, lo, c + d, hi
    z0, z1 = zf - 8, zf + 112
    out = []
    if not hollow:
        out.append(box(bx0, by0, z0, bx1, by1, z1, {"side": "ho_lattice", "bottom": "ho_cumba", "top": color}, align="local", fit="uv"))
    else:
        out.append(box(bx0, by0, z0, bx1, by1, zf, {"top": "ho_floor", "bottom": "ho_cumba", "side": color}))
        out.append(box(bx0, by0, z1 - 8, bx1, by1, z1, {"bottom": "ho_ceil", "side": color, "top": color}))
        wz0, wz1 = zf + 36, zf + 92
        # üç yan duvar: pencere delikleri (önde iki, yanlarda bir)
        if face in ("-y", "+y"):
            fy0, fy1 = (by0, by0 + 8) if face == "-y" else (by1 - 8, by1)
            out += wall_y(fy0, fy1, bx0, bx1, zf, z1 - 8, [(lo + 16, lo + 56, wz0, wz1), (hi - 56, hi - 16, wz0, wz1)],
                          {"side": "ho_lattice", "tb": color}, align="local")
            for sx0, sx1 in ((bx0, bx0 + 8), (bx1 - 8, bx1)):
                out += wall_x(sx0, sx1, by0 + 8, by1 - 8, zf, z1 - 8, [(by0 + 12, by0 + 36, wz0, wz1)] if face == "-y" else [(by1 - 36, by1 - 12, wz0, wz1)],
                              {"side": "ho_lattice", "tb": color}, align="local")
        else:
            fx0, fx1 = (bx0, bx0 + 8) if face == "-x" else (bx1 - 8, bx1)
            out += wall_x(fx0, fx1, by0, by1, zf, z1 - 8, [(lo + 16, lo + 56, wz0, wz1), (hi - 56, hi - 16, wz0, wz1)],
                          {"side": "ho_lattice", "tb": color}, align="local")
            for sy0, sy1 in ((by0, by0 + 8), (by1 - 8, by1)):
                out += wall_y(sy0, sy1, bx0 + 8, bx1 - 8, zf, z1 - 8, [(bx0 + 12, bx0 + 36, wz0, wz1)] if face == "-x" else [(bx1 - 36, bx1 - 12, wz0, wz1)],
                              {"side": "ho_lattice", "tb": color}, align="local")
    # altındaki payanda (kama)
    if face == "-y":
        out.append(Brush([(lo + 8, c, z0), (hi - 8, c, z0), (lo + 8, c, z0 - 40), (hi - 8, c, z0 - 40), (lo + 8, c - d, z0), (hi - 8, c - d, z0)], "ho_wood2"))
    elif face == "+y":
        out.append(Brush([(lo + 8, c, z0), (hi - 8, c, z0), (lo + 8, c, z0 - 40), (hi - 8, c, z0 - 40), (lo + 8, c + d, z0), (hi - 8, c + d, z0)], "ho_wood2"))
    elif face == "-x":
        out.append(Brush([(c, lo + 8, z0), (c, hi - 8, z0), (c, lo + 8, z0 - 40), (c, hi - 8, z0 - 40), (c - d, lo + 8, z0), (c - d, hi - 8, z0)], "ho_wood2"))
    else:
        out.append(Brush([(c, lo + 8, z0), (c, hi - 8, z0), (c, lo + 8, z0 - 40), (c, hi - 8, z0 - 40), (c + d, lo + 8, z0), (c + d, hi - 8, z0)], "ho_wood2"))
    return out


def roof(x0, y0, x1, y1, z, color):
    return [box(x0 - 28, y0 - 28, z - 8, x1 + 28, y1 + 28, z, {"bottom": "ho_eave", "side": "ho_wood2", "top": "ho_tile"}),
            hip_roof(x0 - 28, y0 - 28, x1 + 28, y1 + 28, z, z + 80, {"slope": "ho_tile", "side": "ho_tile", "bottom": "ho_eave"})]


def facade_house(x0, y0, x1, y1, base, faces, floors=2, color=None, door_face=None, cumbas=(), shop=None, sign=None, buried=()):
    """Girilmeyen ev: dolu blok + kabartma pencere/kapı + cumba + kiremit çatı.
    faces: pencere konan yüzler. cumbas: (face, center) listesi. shop: (face, center) vitrin."""
    color = color or next_color()
    H = 128 * floors
    group("ev", shadow=True)
    W(box(x0, y0, base, x1, y1, base + H, {"side": color, "tb": color}))
    D(box(x0 - 2, y0 - 2, base, x1 + 2, y1 + 2, base + 40, "ho_stone"))
    for f in faces:
        if f not in buried:
            D(windows_on(f, x0, y0, x1, y1, base + 56))
        if floors == 2:
            D(windows_on(f, x0, y0, x1, y1, base + 168))
    if door_face:
        c, a0, a1 = face_span(door_face, x0, y0, x1, y1)
        m = (a0 + a1) / 2 + (-48 if (a1 - a0) > 200 else 0)
        D(relief(door_face, c, m - 32, m + 32, base, base + 104, "ho_door"))
    if shop:
        f, m = shop
        c, a0, a1 = face_span(f, x0, y0, x1, y1)
        D(relief(f, c, m - 64, m + 64, base + 8, base + 120, "ho_shop", 6))
        # tente
        if f == "-y":
            D(Brush([(m - 80, c, base + 124), (m + 80, c, base + 124), (m - 80, c - 56, base + 100), (m + 80, c - 56, base + 100),
                     (m - 80, c, base + 128), (m + 80, c, base + 128), (m - 80, c - 56, base + 104), (m + 80, c - 56, base + 104)], "ho_awning"))
        elif f == "+y":
            D(Brush([(m - 80, c, base + 124), (m + 80, c, base + 124), (m - 80, c + 56, base + 100), (m + 80, c + 56, base + 100),
                     (m - 80, c, base + 128), (m + 80, c, base + 128), (m - 80, c + 56, base + 104), (m + 80, c + 56, base + 104)], "ho_awning2"))
        elif f == "-x":
            D(Brush([(c, m - 80, base + 124), (c, m + 80, base + 124), (c - 56, m - 80, base + 100), (c - 56, m + 80, base + 100),
                     (c, m - 80, base + 128), (c, m + 80, base + 128), (c - 56, m - 80, base + 104), (c - 56, m + 80, base + 104)], "ho_awning"))
        else:
            D(Brush([(c, m - 80, base + 124), (c, m + 80, base + 124), (c + 56, m - 80, base + 100), (c + 56, m + 80, base + 100),
                     (c, m - 80, base + 128), (c, m + 80, base + 128), (c + 56, m - 80, base + 104), (c + 56, m + 80, base + 104)], "ho_awning"))
    if sign:
        f, m, tex, sw, sh = sign
        c, a0, a1 = face_span(f, x0, y0, x1, y1)
        D(relief(f, c, m - sw / 2, m + sw / 2, base + 132, base + 132 + sh, tex, 6))
    if floors == 2:
        for f, m in cumbas:
            D(cumba(f, x0, y0, x1, y1, base + 136, m, color=color))
    W(roof(x0, y0, x1, y1, base + H, color))


def passable_house(x0, y0, x1, y1, base, doors, color=None, cumbas=(), windows=(), two_level=False, stair_side="-x", upper_doors=(), lamp=True, buried=()):
    """Girilen ev. doors: (face, center) zemin kat kapıları (z=base). upper_doors: (face, center) üst kat kapıları
    (z=base+128; dışarıdaki yukarı kot için). two_level: iç merdiven + üst kat. cumbas: (face, center) içi boş cumba (üst kat).
    stair_side: merdivenin dayandığı duvar ('-x' batı duvarı boyunca y yönünde çıkar; '+x' doğu duvarı)."""
    color = color or next_color()
    T = 16
    H = 256 if two_level else 128
    ztop = base + H
    group("ev", shadow=True)
    walls = {"-y": [], "+y": [], "-x": [], "+x": []}
    for f, m in doors:
        walls[f].append((m - 32, m + 32, base, base + 96))
    for f, m in upper_doors:
        walls[f].append((m - 32, m + 32, base + 128, base + 224))
    for f, m in cumbas:
        walls[f].append((m - 56, m + 56, base + 136, base + 240))
    tex = lambda out: {out: color, "side": "ho_inner", "tb": color}
    W(wall_y(y0, y0 + T, x0 + T, x1 - T, base, ztop, walls["-y"], tex("-y")))
    W(wall_y(y1 - T, y1, x0 + T, x1 - T, base, ztop, walls["+y"], tex("+y")))
    W(wall_x(x0, x0 + T, y0, y1, base, ztop, walls["-x"], tex("-x")))
    W(wall_x(x1 - T, x1, y0, y1, base, ztop, walls["+x"], tex("+x")))
    D(box(x0 - 2, y0 - 2, base, x1 + 2, y1 + 2, base + 40, "ho_stone"))
    # zemin, tavan
    W(box(x0 + T, y0 + T, base, x1 - T, y1 - T, base + 2, {"top": "ho_floor", "side": "ho_floor", "bottom": "ho_floor"}))
    for f in windows:
        if f not in buried:
            D(windows_on(f, x0, y0, x1, y1, base + 56))
        if two_level:
            D(windows_on(f, x0, y0, x1, y1, base + 168))
    ix0, iy0, ix1, iy1 = x0 + T, y0 + T, x1 - T, y1 - T
    if two_level:
        # ara kat döşemesi (merdiven boşluğu oyulur), merdiven batı/doğu duvarı boyunca kuzeye çıkar
        sx0, sx1 = (ix0, ix0 + 56) if stair_side == "-x" else (ix1 - 56, ix1)
        sd = min(24, int((iy1 - iy0 - 24) / 8))
        run = 8 * sd
        sy0 = iy1 - run - 8
        slab = [box(ix0, iy0, base + 128, ix1, iy1, base + 136, {"top": "ho_floor", "bottom": "ho_ceil", "side": "ho_wood2"})]
        slab = carve(slab, (sx0 - 1, sy0 - 40, base + 127, sx1 + 1, iy1 + 1, base + 137))
        W(slab)
        for k in range(1, 9):
            W(box(sx0, sy0 + sd * (k - 1), base, sx1, sy0 + sd * k, base + 16 * k, {"top": "ho_wood", "side": "ho_wood2"}))
        W(box(sx0, iy1 - 8, base, sx1, iy1, base + 136, "ho_wood2"))           # sahanlık dolgusu
        # merdiven korkuluğu (üst katta boşluk kenarı)
        rx = sx1 if stair_side == "-x" else sx0 - 6
        W(box(rx, sy0 - 40, base + 136, rx + 6, sy0 + 112, base + 176, "ho_wood2"))
        # ara bölme duvarı (her katta), kapı boşluklu; merdiven sütunu dışarıda, dar evde bölme yok
        pm = (iy0 + iy1) / 2
        px0, px1 = (sx1, ix1) if stair_side == "-x" else (ix0, sx0)
        if px1 - px0 >= 112 and iy1 - iy0 >= 256:
            dm = (px0 + px1) / 2
            W(wall_y(pm - 8, pm + 8, px0, px1, base, base + 128, [(dm - 32, dm + 32, base, base + 96)], {"side": "ho_inner", "tb": "ho_inner"}))
            W(wall_y(pm - 8, pm + 8, px0, px1, base + 136, base + 256, [(dm - 32, dm + 32, base + 136, base + 232)], {"side": "ho_inner", "tb": "ho_inner"}))
        W(box(ix0, iy0, base + 256, ix1, iy1, base + 264, {"bottom": "ho_ceil", "side": "ho_wood2", "top": "ho_wood2"}))
        for f, m in cumbas:
            D(cumba(f, x0, y0, x1, y1, base + 136, m, hollow=True, color=color))
        if lamp:
            light(((ix0 + ix1) / 2, (iy0 + pm) / 2, base + 110), (255, 236, 200), 200)
            light(((ix0 + ix1) / 2, (pm + iy1) / 2, base + 110), (255, 236, 200), 200)
            light(((ix0 + ix1) / 2, (iy0 + pm) / 2, base + 240), (255, 236, 200), 200)
            light(((ix0 + ix1) / 2, (pm + iy1) / 2, base + 240), (255, 236, 200), 200)
    else:
        W(box(ix0, iy0, base + 128, ix1, iy1, base + 136, {"bottom": "ho_ceil", "side": "ho_wood2", "top": "ho_wood2"}))
        pm = (iy0 + iy1) / 2
        if iy1 - iy0 > 200:
            W(wall_y(pm - 8, pm + 8, ix0, ix1, base, base + 128, [((ix0 + ix1) / 2 - 32, (ix0 + ix1) / 2 + 32, base, base + 96)], {"side": "ho_inner", "tb": "ho_inner"}))
        if lamp:
            light(((ix0 + ix1) / 2, (iy0 + pm) / 2, base + 110), (255, 236, 200), 200)
            light(((ix0 + ix1) / 2, (pm + iy1) / 2, base + 110), (255, 236, 200), 200)
    # kilim ve eşya
    D(box(ix0 + 24, iy0 + 24, base + 2, ix0 + 88, iy0 + 120, base + 3, "ho_kilim", align="local", fit="uv"))
    W(roof(x0, y0, x1, y1, ztop, color))


def house_row(face, c, a0, a1, base, depth, floors=2, min_w=160, max_w=256, seed=0, shops=False):
    """Sokak kenarı boyunca art arda cepheli evler. face: evlerin baktığı yön (sokağa bakan yüzün normali)."""
    rs = np.random.default_rng(seed)
    a = a0
    i = 0
    while a1 - a >= min_w:
        w = float(rs.integers(min_w, max_w + 1))
        if a1 - (a + w) < min_w:
            w = a1 - a
        b = a + w
        if face == "-y":
            x0, y0, x1, y1 = a, c, b, c + depth
        elif face == "+y":
            x0, y0, x1, y1 = a, c - depth, b, c
        elif face == "-x":
            x0, y0, x1, y1 = c, a, c + depth, b
        else:
            x0, y0, x1, y1 = c - depth, a, c, b
        m = (a + b) / 2
        cum = [(face, m + (w / 4 if w > 220 else 0))] if (floors == 2 and i % 2 == 0) else []
        shop = (face, m - w / 4) if (shops and i % 3 == 1 and w > 200) else None
        facade_house(x0, y0, x1, y1, base, [face], floors, door_face=face if not shop else None, cumbas=cum, shop=shop)
        a = b
        i += 1


def tree(x, y, base, h=150, r=90):
    """Gövde + yaprak kümesi: iki çapraz dikey levha ve bir yatay levha ({ şeffaf doku), gölge atar."""
    group("agac")
    D(prism(ngon(x, y, 10, 8), base, base + h, "ho_bark", align="local"))
    zc = base + h
    leaves = [box(x - r, y - 2, zc - 70, x + r, y + 2, zc + 70, "{ho_leaves", align="local"),
              box(x - 2, y - r, zc - 70, x + 2, y + r, zc + 70, "{ho_leaves", align="local"),
              box(x - r * 0.8, y - r * 0.8, zc - 2, x + r * 0.8, y + r * 0.8, zc + 2, "{ho_leaves", align="local")]
    ent("func_illusionary", brushes=leaves, rendermode=4, renderamt=255, zhlt_lightflags=2)


def bench(x, y, base, along="x"):
    group("bank")
    if along == "x":
        D(box(x - 48, y - 12, base + 20, x + 48, y + 12, base + 28, "ho_bench"), box(x - 48, y + 12, base + 28, x + 48, y + 20, base + 60, "ho_bench"),
          box(x - 44, y - 12, base, x - 36, y + 20, base + 20, "ho_metal"), box(x + 36, y - 12, base, x + 44, y + 20, base + 20, "ho_metal"))
    else:
        D(box(x - 12, y - 48, base + 20, x + 12, y + 48, base + 28, "ho_bench"), box(x + 12, y - 48, base + 28, x + 20, y + 48, base + 60, "ho_bench"),
          box(x - 12, y - 44, base, x + 20, y - 36, base + 20, "ho_metal"), box(x - 12, y + 36, base, x + 20, y + 44, base + 20, "ho_metal"))


def lamp_post(x, y, base, h=176):
    group("lamba")
    D(box(x - 4, y - 4, base, x + 4, y + 4, base + h, "ho_metal"),
      box(x - 12, y - 12, base + h, x + 12, y + 12, base + h + 24, {"side": "ho_lamp", "tb": "ho_metal"}),
      pyramid(ngon(x, y, 14, 4, math.pi / 4), base + h + 24, (x, y, base + h + 40), "ho_metal"))
    light((x, y, base + h - 6), (255, 236, 200), 120)


def fountain(x, y, base, r=72):
    group("cesme")
    D(prism(ngon(x, y, r, 8), base, base + 40, {"side": "ho_ashlar", "top": "ho_ashlar"}, align="local"))
    ent("func_water", brushes=[prism(ngon(x, y, r - 12, 8), base + 16, base + 36, "!ho_water")], skin=-3, wavehight=0)
    D(prism(ngon(x, y, 12, 8), base, base + 80, "ho_ashlar"), prism(ngon(x, y, 24, 8), base + 80, base + 88, "ho_ashlar"))
    sound((x, y, base + 60), "hamamonu/cesme.wav", 6, 2)


def umbrella(x, y, base):
    group("semsiye")
    D(box(x - 3, y - 3, base, x + 3, y + 3, base + 120, "ho_metal"),
      pyramid(ngon(x, y, 72, 8), base + 100, (x, y, base + 130), "ho_awning"),
      box(x - 32, y - 32, base + 44, x + 32, y + 32, base + 48, "ho_wood"), box(x - 2, y - 2, base, x + 2, y + 2, base + 44, "ho_metal"))
    for dx, dy in ((-50, 0), (50, 0), (0, -50), (0, 50)):
        D(box(x + dx - 12, y + dy - 12, base + 26, x + dx + 12, y + dy + 12, base + 30, "ho_wood"),
          box(x + dx - 2, y + dy - 2, base, x + dx + 2, y + dy + 2, base + 26, "ho_metal"))


def planter(x0, y0, x1, y1, base, h=32):
    group("saksi")
    D(box(x0, y0, base, x1, y1, base + h, {"side": "ho_gwall", "top": "ho_grass"}),
      box(x0 + 16, y0 + 16, base + h, x1 - 16, y1 - 16, base + h + 28, "ho_hedge", align="local"))


# ================================================================ DÜNYA
X0, X1, Y0, Y1, Z0, Z1 = -1792, 1536, -1152, 1536, -128, 1088
T = 32
W(box(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T, "sky"),
  box(X0 - T, Y0 - T, Z0 - T, X1 + T, Y1 + T, Z0, "ho_dirt"),
  box(X0 - T, Y0, Z0, X0, Y1, Z1, "sky"),
  box(X1, Y0, Z0, X1 + T, Y1, Z1, "sky"),
  box(X0 - T, Y0 - T, Z0, X1 + T, Y0, Z1, "sky"),
  box(X0 - T, Y1, Z0, X1 + T, Y1 + T, Z1, "sky"))

LT = {"cobble": {"top": "ho_cobble", "side": "ho_stone"}, "paving": {"top": "ho_paving", "side": "ho_stone"},
      "paving2": {"top": "ho_paving2", "side": "ho_stone"}, "dirt": {"top": "ho_dirt", "side": "ho_stone"},
      "grass": {"top": "ho_grass", "side": "ho_stone"}}
# alt kot: taban + bölge kaplamaları (z -8..0)
W(box(X0, Y0, Z0, X1, Y1, -8, "ho_dirt"))
L_REGIONS = [
    (X0, Y0, -1408, 384, "dirt"),                          # batı şerit
    (-1408, -384, -640, 384, "paving"),                    # B: Saat Kulesi Meydanı
    (-640, -608, -512, 256, "cobble"),                     # Hamam Sokağı
    (-1408, -768, 1536, -608, "cobble"),                   # güney sokak
    (-1408, -1152, -768, -768, "paving2"),                 # T meydanı (aşağı mahalle)
    (-768, -1152, 1536, -768, "dirt"),                     # güney evler
    (-1408, -608, -1100, -384, "dirt"), (-950, -608, -640, -384, "dirt"),   # güneybatı blok
    (-1100, -608, -950, -384, "cobble"),                   # T sokağı (B'ye)
    (-512, -608, -128, -192, "dirt"), (-128, -608, -32, -192, "cobble"), (-32, -608, 256, -192, "dirt"),
    (-512, -192, 256, -64, "cobble"),                      # hamam avlusu
    (-512, -64, 256, 256, "dirt"),                         # H1 sırası
    (256, -608, 352, 256, "cobble"),                       # merdivenli sokak
    (352, -608, 512, 256, "dirt"),                         # H3
]
GROUND = [box(x0, y0, -8, x1, y1, 0, LT[t]) for x0, y0, x1, y1, t in L_REGIONS if x1 > x0]
# üst kot plakaları (z 0..128), yanları taş istinat duvarı
U_REGIONS = [
    (X0, 384, -1100, 1536, "dirt"), (-1100, 384, -980, 640, "cobble"), (-980, 384, -640, 832, "dirt"),
    (-980, 960, -640, 1536, "dirt"),
    (-1100, 640, -980, 832, "cobble"), (-1100, 832, 512, 960, "cobble"), (-1100, 960, -980, 1536, "dirt"),
    (-640, 256, 512, 448, "cobble"),                        # ana sokak
    (-640, 448, 512, 832, "dirt"), (-640, 960, 512, 1536, "dirt"),
    (512, -608, 1408, 640, "paving"),                       # A: Hacettepe Önü
    (512, 640, 640, 1024, "paving"), (1408, 640, 1536, 1024, "paving"),
    (640, 640, 1408, 1024, "dirt"), (512, 1024, 1536, 1536, "paving2"), (1408, -608, 1536, 640, "dirt"),
]
UPPER = [box(x0, y0, 0, x1, y1, 128, LT[t]) for x0, y0, x1, y1, t in U_REGIONS if x1 > x0]
# merdiven oyukları: Kule yokuşu (kuzey sokaktan B'ye) ve Hacettepe güney merdiveni (A'dan güney sokağa)
UPPER = carve(UPPER, (-1100, 384, -1, -980, 640, 129))
UPPER = carve(UPPER, (896, -608, -1, 1024, -352, 129))
for k in range(1, 9):
    W(box(-1100, 384 + 32 * (k - 1), 0, -980, 384 + 32 * k, 16 * k, {"top": "ho_paving", "side": "ho_stone"}))
    W(box(896, -608 + 32 * (k - 1), 0, 1024, -608 + 32 * k, 16 * k, {"top": "ho_paving", "side": "ho_stone"}))
W(box(-1104, 384, 0, -1100, 640, 150, "ho_stone"), box(-980, 384, 0, -976, 640, 150, "ho_stone"))   # yokuş yan duvarları
W(box(892, -608, 0, 896, -352, 150, "ho_stone"), box(1024, -608, 0, 1028, -352, 150, "ho_stone"))
# Büyük merdiven: ana sokaktan B'ye (batıya iner), güney kenarında alçak korkuluk
for k in range(1, 9):
    W(box(-640 - 32 * k, 256, 0, -640 - 32 * (k - 1), 384, 128 - 16 * (k - 1), {"top": "ho_ashlar", "side": "ho_ashlar2"}))
W(box(-896, 248, 0, -640, 256, 150, {"side": "ho_ashlar2", "top": "ho_ashlar"}))
# merdivenli sokak (alley 2): y 0..256 arasında kuzeye çıkar
for k in range(1, 9):
    W(box(256, 32 * (k - 1), 0, 352, 32 * k, 16 * k, {"top": "ho_cobble", "side": "ho_stone"}))

# ================================================================ ÇEVRE DUVARLARI
PW = {"side": "ho_gwall", "top": "ho_ashlar", "bottom": "ho_gwall"}
W(box(X0, Y0, 0, X1, Y0 + 24, 200, PW), box(X0, Y1 - 24, 128, X1, Y1, 328, PW),
  box(X0, Y0, 0, X0 + 24, Y1, 328, PW), box(X1 - 24, Y0, 0, X1, Y1, 328, PW))
W(box(-1408, -1152, 0, -768, -1120, 56, PW))                            # T meydanı güney bahçe duvarı

# ================================================================ B: SAAT KULESİ MEYDANI (z=0)
TX, TY = -1024, 0
W(box(TX - 72, TY - 72, 0, TX + 72, TY + 72, 32, "ho_ashlar"),
  box(TX - 56, TY - 56, 32, TX + 56, TY + 56, 556, "ho_ashlar2", align="local"))
group("kule", shadow=True)
D(box(TX - 64, TY - 64, 420, TX + 64, TY + 64, 436, "ho_ashlar"),
  box(TX - 68, TY - 68, 540, TX + 68, TY + 68, 556, "ho_ashlar"),
  pyramid(ngon(TX, TY, 80, 4, math.pi / 4), 556, (TX, TY, 650), "ho_dome"))
for f, c in (("-x", TX - 56), ("+x", TX + 56), ("-y", TY - 56), ("+y", TY + 56)):
    if f in ("-x", "+x"):
        D(relief(f, c, TY - 40, TY + 40, 448, 528, "ho_clock"))
    else:
        D(relief(f, c, TX - 40, TX + 40, 448, 528, "ho_clock"))
D(prism(ngon(TX, TY, 128, 8), 0, 24, {"side": "ho_ashlar", "top": "ho_bench"}, align="local"))     # çevre oturma halkası
D(relief("-y", TY - 72, TX - 32, TX + 32, 40, 104, "ho_signb"))
fountain(-1200, 220, 0)
for x, y in ((-1350, -300), (-700, 300), (-1350, 300)):
    tree(x, y, 0)
# çay bahçesi (güneybatı köşe): şemsiyeler, masalar; çit
for x, y in ((-1300, -200), (-1180, -200), (-1300, -320), (-1180, -320)):
    umbrella(x, y, 0)
ent("func_wall", brushes=[box(-1380, -120, 0, -1120, -116, 40, "{ho_fence", align="local"),
                          box(-1120, -380, 0, -1116, -116, 40, "{ho_fence", align="local")], rendermode=4, renderamt=255)
for x, y in ((-760, -300), (-760, 300), (-1340, 60), (-700, 0)):
    lamp_post(x, y, 0)
bench(-1250, 350, 0); bench(-800, -360, 0)
sound((-1240, -260, 80), "hamamonu/kalabalik.wav", 6, 2)
sound((-1000, 0, 200), "hamamonu/kus.wav", 6, 8)

# B'nin batısı: cepheli evler (doğuya bakar), T meydanının batısı
house_row("+x", -1408, -384, 384, 0, 200, floors=2, seed=1, shops=True)
house_row("+x", -1408, -1120, -768, 0, 200, floors=2, seed=2)
# güneybatı blok (B ile güney sokak arası): cepheler + T sokağı + içinden geçilen SW1
facade_house(-1408, -608, -1250, -384, 0, ["-y", "+y"], 2, cumbas=[("+y", -1330)], door_face="+y")
facade_house(-1250, -608, -1100, -384, 0, ["-y", "+y"], 2, door_face="-y", shop=("-y", -1175))
facade_house(-950, -608, -820, -384, 0, ["-y", "+y"], 2, cumbas=[("-y", -885)], door_face="+y")
passable_house(-820, -608, -640, -384, 0, [("-y", -730), ("+y", -730)], windows=["-y", "+y"], two_level=True, cumbas=[("+y", -730)], stair_side="-x")

# ================================================================ T MEYDANI (aşağı mahalle)
fountain(-860, -960, 0, 60)
tree(-900, -1040, 0); tree(-1300, -1040, 0)
bench(-1000, -1080, 0); bench(-1180, -1080, 0)
lamp_post(-820, -820, 0); lamp_post(-1360, -820, 0)
# güney sokağın güneyi: cephe sırası (kuzeye bakar)
house_row("+y", -768, -768, 1536, 0, 192, floors=2, seed=3, shops=True)
W(box(-768, -1152, 0, 1536, -960, 40, "ho_gwall"))                       # arka bahçe duvarları dolgusu

# ================================================================ ORTA BLOK: HAMAM, H2, AVLU, H1 SIRASI, H3
# Karacabey Hamamı: x -512..-128, y -608..-192 ; kapılar güney (sokak) ve kuzey (avlu)
HX0, HY0, HX1, HY1 = -512, -608, -128, -192
HT = {"side": "ho_ashlar2", "tb": "ho_ashlar2"}
W(wall_y(HY0, HY0 + 32, HX0 + 32, HX1 - 32, 0, 256, [(-352, -288, 0, 112)], HT))
W(wall_y(HY1 - 32, HY1, HX0 + 32, HX1 - 32, 0, 256, [(-352, -288, 0, 112)], HT))
W(box(HX0, HY0, 0, HX0 + 32, HY1, 256, HT), box(HX1 - 32, HY0, 0, HX1, HY1, 256, HT))
W(box(HX0 + 32, HY0 + 32, 0, HX1 - 32, HY1 - 32, 2, {"top": "ho_marble", "side": "ho_marble"}),
  box(HX0, HY0, 256, HX1, HY1, 272, {"bottom": "ho_marble", "side": "ho_ashlar2", "top": "ho_dome"}))
GX, GY = -320, -400
group("hamam", shadow=True)
for r, z0, z1 in ((150, 272, 310), (122, 310, 344), (90, 344, 370), (52, 370, 390), (18, 390, 420)):
    D(prism(ngon(GX, GY, r, 16), z0, z1, "ho_dome", align="local"))
for cx_, cy_ in ((-440, -540), (-200, -540), (-440, -260), (-200, -260)):
    D(prism(ngon(cx_, cy_, 44, 12), 272, 300, "ho_dome", align="local"), prism(ngon(cx_, cy_, 24, 12), 300, 312, "ho_dome", align="local"))
D(prism(ngon(GX, GY, 72, 8), 0, 28, {"side": "ho_marble", "top": "ho_marble"}))                 # göbek taşı
for cx_, cy_ in ((-440, -540), (-200, -540), (-440, -260), (-200, -260)):                         # kurnalar
    D(box(cx_ - 24, cy_ - 16, 0, cx_ + 24, cy_ + 16, 36, "ho_marble"))
# iç bölme duvarları: dört halvet (köşe odaları), kapı boşluklu
W(wall_x(-392, -376, HY0 + 32, -480, 0, 256, [(-560, -500, 0, 112)], {"side": "ho_ashlar2", "tb": "ho_ashlar2"}))
W(wall_x(-264, -248, HY0 + 32, -480, 0, 256, [(-560, -500, 0, 112)], {"side": "ho_ashlar2", "tb": "ho_ashlar2"}))
W(wall_x(-392, -376, -320, HY1 - 32, 0, 256, [(-300, -240, 0, 112)], {"side": "ho_ashlar2", "tb": "ho_ashlar2"}))
W(wall_x(-264, -248, -320, HY1 - 32, 0, 256, [(-300, -240, 0, 112)], {"side": "ho_ashlar2", "tb": "ho_ashlar2"}))
D(relief("-y", HY0, -448, -192, 132, 196, "ho_s_hamam", 6))
for pos in ((-320, -400, 200), (-440, -540, 120), (-200, -260, 120), (-320, -560, 120), (-320, -230, 120)):
    light(pos, (255, 240, 210), 140)
sound((-320, -400, 100), "hamamonu/damla.wav", 6, 2)

# H2: güney sokak <-> avlu, üst katta sokağa bakan içi boş cumba (çıkmaz oda)
passable_house(-32, -608, 256, -192, 0, [("-y", 112), ("+y", 112)], color="ho_pl_y", windows=["-y", "+y"], two_level=True,
               cumbas=[("-y", 160)], stair_side="+x")
group('tabela'); D(relief("-y", -608, -16, 112, 136, 168, "ho_s_bakir", 6))
# H1 sırası: avlu (z=0) <-> ana sokak (z=128)
passable_house(-512, -64, -320, 256, 0, [("-y", -416)], upper_doors=[("+y", -416)], color="ho_pl_w", windows=["-y", "+y"], two_level=True,
               cumbas=[("-y", -400)], stair_side="-x", buried=["+y"])
facade_house(-320, -64, -128, 256, 0, ["-y", "+y"], 2, color="ho_pl_p", cumbas=[("-y", -224)], door_face="-y", buried=["+y"])
passable_house(-128, -64, 64, 256, 0, [("-y", -32)], upper_doors=[("+y", -32)], color="ho_pl_b", windows=["-y", "+y"], two_level=True,
               cumbas=[("-y", 0)], stair_side="+x", buried=["+y"])
passable_house(64, -64, 256, 256, 0, [("-y", 160), ("+x", -24)], upper_doors=[("+y", 160)], color="ho_pl_g", windows=["-y", "+y"], two_level=True,
               cumbas=[("-y", 160)], stair_side="-x", buried=["+y"])
# H3: merdivenli sokak (z=0) <-> A (z=128); kuzey ve güney parçaları cephe
facade_house(352, -608, 512, -300, 0, ["-x", "+x"], 2, color="ho_pl_w", cumbas=[("-x", -450)], buried=["+x"])
passable_house(352, -300, 512, -60, 0, [("-x", -180)], upper_doors=[("+x", -180)], color="ho_pl_y", windows=["-x", "+x"], two_level=True, stair_side="-x", buried=["+x"])
facade_house(352, -60, 512, 256, 0, ["-x", "+x", "+y"], 2, color="ho_pl_p", cumbas=[("-x", 100)], buried=["+x", "+y"])
group('tabela'); D(relief("+x", 512, -640, -384, 132, 164, "ho_s_hacetyol", 6))
# avlu eşyaları
tree(-100, -128, 0, 190, 70)
bench(60, -100, 0)
lamp_post(-480, -100, 0); lamp_post(200, -176, 0)
sound((-128, -128, 100), "hamamonu/kus.wav", 5, 4)
# Hamam Sokağı & güney sokak eşyaları
lamp_post(-560, -560, 0); lamp_post(-560, 100, 0); lamp_post(-80, -640, 0); lamp_post(600, -690, 0); lamp_post(1200, -690, 0)
group("sokak"); D(box(-620, -200, 0, -596, -100, 56, "ho_ashlar"), box(-620, -100, 0, -596, -40, 56, "ho_ashlar"))   # çiçek saksıları
D(box(-600, -560, 0, -536, -520, 48, "ho_wood"), box(-600, -520, 0, -536, -500, 48, "ho_wood"))       # sandıklar

# ================================================================ ANA SOKAK ve KUZEY BLOK (z=128)
NB = 128
# kuzey blok: ara sokaklar x -448..-352 ve x 160..256 ; evler y 448..832
facade_house(-640, 448, -448, 832, NB, ["-y", "+y"], 2, color="ho_pl_b", cumbas=[("-y", -544)], door_face="-y")
passable_house(-352, 448, -96, 832, NB, [("-y", -224), ("+y", -224)], color="ho_pl_w", windows=["-y", "+y"], two_level=True,
               cumbas=[("-y", -224)], stair_side="+x")
facade_house(-96, 448, 160, 832, NB, ["-y", "+y"], 2, color="ho_pl_y", cumbas=[("-y", 32), ("+y", 32)], door_face="-y", shop=("-y", -40))
passable_house(256, 448, 512, 832, NB, [("-y", 384), ("+y", 384)], color="ho_pl_p", windows=["-y", "+y"], two_level=False)
group('tabela'); D(relief("-y", 448, -600, -344, 132 + NB, 164 + NB, "ho_s_kule", 6))
group('tabela'); D(relief("-y", 448, 20, 148, 136 + NB, 168 + NB, "ho_s_bakir", 6))
# ana sokak eşyaları: lambalar, saksılar, kilim asılı duvar
for x in (-500, -200, 100, 400):
    lamp_post(x, 440, NB)
group("sokak"); D(box(-300, 260, NB, -236, 300, NB + 56, "ho_ashlar"), box(300, 260, NB, 364, 300, NB + 56, "ho_ashlar"))
group('tabela'); D(relief("+y", 256, -300, -172, NB + 24, NB + 120, "ho_kilim", 6))            # H1b cephesinde asılı kilim
group('tabela'); D(relief("+y", 256, 120, 184, NB + 150, NB + 246, "ho_poster", 6))
# kuzeybatı blok: B'ye yukarıdan bakan cumbalı evler (güneye bakar), kule yokuşu x -1100..-980
facade_house(-1408, 384, -1250, 640, NB, ["-y"], 2, color="ho_pl_y", cumbas=[("-y", -1330)])
facade_house(-1250, 384, -1100, 640, NB, ["-y"], 2, color="ho_pl_w", cumbas=[("-y", -1175)], door_face="-y")
facade_house(-980, 384, -800, 640, NB, ["-y", "-x"], 2, color="ho_pl_p", cumbas=[("-y", -890)])
facade_house(-800, 384, -640, 640, NB, ["-y"], 2, color="ho_pl_g", cumbas=[("-y", -720)], door_face="-y")
W(box(-1408, 640, NB, -1100, 832, NB + 256, "ho_pl_w"), roof(-1408, 640, -1100, 832, NB + 256, "ho_pl_w"))     # dolgu ev
W(box(-980, 640, NB, -640, 832, NB + 256, "ho_pl_b"), roof(-980, 640, -640, 832, NB + 256, "ho_pl_b"))
group("ev"); D(windows_on("+y", -980, 640, -640, 832, NB + 168), windows_on("+y", -1408, 640, -1100, 832, NB + 168))
# kuzey sokağın kuzeyi: cephe sırası (güneye bakar), batı ucu dolgu
house_row("-y", 960, -1100, 512, NB, 224, floors=2, seed=4, shops=True)
house_row("+x", -1408, 384, 1536, NB, 200, floors=2, seed=6)
W(box(X0 + 24, 384, NB, -1608, 1536, NB + 200, "ho_gwall"))
W(box(-1408, 960, NB, -1100, 1536, NB + 256, "ho_pl_g"), roof(-1408, 960, -1100, 1536, NB + 256, "ho_pl_g"))
W(box(-1100, 1184, NB, 512, 1536, NB + 200, "ho_gwall"))
lamp_post(-1040, 900, NB); lamp_post(-400, 900, NB); lamp_post(200, 900, NB); lamp_post(560, 980, NB)
sound((-300, 900, NB + 100), "hamamonu/kus.wav", 5, 4)
group('tabela'); D(relief("-y", 384, -1250, -1100, NB + 60, NB + 92, "ho_s_kule", 6))

# ================================================================ A: HACETTEPE ÖNÜ (z=128) ve HACETTEPE
AX0, AX1 = 512, 1408
# Hacettepe binası: x 512..1408, y 640..1024, z 128..512 ; lobi x 832..1088 geçilir
HB = {"side": "ho_hacet", "tb": "ho_concrete"}
W(box(AX0, 640, NB, 832, 1024, NB + 384, HB), box(1088, 640, NB, AX1, 1024, NB + 384, HB))
W(wall_y(640, 656, 832, 1088, NB, NB + 384, [(896, 1024, NB, NB + 160)], {"side": "ho_glass", "tb": "ho_concrete"}))
W(wall_y(1008, 1024, 832, 1088, NB, NB + 384, [(896, 1024, NB, NB + 160)], {"side": "ho_glass", "tb": "ho_concrete"}))
W(box(832, 656, NB + 256, 1088, 1008, NB + 384, {"bottom": "ho_concrete", "side": "ho_concrete", "top": "ho_concrete"}))
W(box(832, 656, NB, 1088, 1008, NB + 2, {"top": "ho_marble", "side": "ho_marble"}))
group("hacettepe", shadow=True)
for cx_ in (880, 1040):
    for cy_ in (740, 924):
        D(box(cx_ - 16, cy_ - 16, NB, cx_ + 16, cy_ + 16, NB + 256, "ho_concrete"))
D(box(900, 800, NB, 1020, 864, NB + 44, "ho_wood2"))                                     # danışma bankosu
D(box(AX0, 1024, NB + 384, AX1, 1040, NB + 400, "ho_concrete"), box(AX0, 624, NB + 384, AX1, 640, NB + 400, "ho_concrete"))
group('tabela'); D(relief("-y", 640, 704, 1216, NB + 176, NB + 240, "ho_s_hacet", 8))
# portik: sundurma ve kolonlar
D(box(800, 560, NB + 168, 1120, 640, NB + 184, "ho_concrete"))
for cx_ in (816, 1104):
    D(box(cx_ - 12, 572, NB, cx_ + 12, 596, NB + 168, "ho_concrete"))
for pos in ((960, 760, NB + 220), (960, 920, NB + 220), (880, 840, NB + 220), (1040, 840, NB + 220)):
    light(pos, (235, 240, 255), 180)
sound((960, 840, NB + 120), "hamamonu/ugultu.wav", 4, 2)
# A meydanı eşyaları: ağaçlar, banklar, saksılar, anıt kaidesi, otobüs durağı, büfe
for x, y in ((620, -480), (620, 380), (1300, -480), (1300, 360), (780, 120), (1140, 120)):
    tree(x, y, NB, 210, 100)
group("anit", shadow=True)
D(box(912, -48, NB, 1008, 48, NB + 80, "ho_ashlar2"), box(928, -32, NB + 80, 992, 32, NB + 96, "ho_ashlar"),
  box(944, -16, NB + 96, 976, 16, NB + 176, "ho_ashlar2"))                                   # anıt
planter(540, -260, 700, -140, NB); planter(1220, -260, 1380, -140, NB); planter(540, 180, 700, 300, NB)
bench(960, -200, NB); bench(960, 200, NB); bench(760, -120, NB, "y"); bench(1160, -120, NB, "y")
# otobüs durağı (güneydoğu)
group("durak", shadow=True)
D(box(1200, -560, NB, 1360, -552, NB + 128, "ho_glass"), box(1192, -568, NB + 128, 1368, -488, NB + 136, "ho_metal"),
  box(1200, -568, NB, 1208, -488, NB + 128, "ho_metal"), box(1352, -568, NB, 1360, -488, NB + 128, "ho_metal"),
  box(1216, -552, NB + 24, 1344, -528, NB + 32, "ho_bench"), relief("+y", -552, 1256, 1304, NB + 60, NB + 108, "ho_bus"))
# büfe
group("bufe", shadow=True)
D(box(1150, 200, NB, 1278, 296, NB + 112, {"side": "ho_metal", "tb": "ho_metal"}), relief("-y", 200, 1166, 1262, NB + 20, NB + 100, "ho_shop", 6),
  Brush([(1140, 200, NB + 108), (1288, 200, NB + 108), (1140, 150, NB + 92), (1288, 150, NB + 92),
         (1140, 200, NB + 112), (1288, 200, NB + 112), (1140, 150, NB + 96), (1288, 150, NB + 96)], "ho_awning2"))
group('tabela'); D(relief("-x", 1408, -200, -136, NB + 40, NB + 104, "ho_signa"))
lamp_post(700, -560, NB); lamp_post(700, 560, NB); lamp_post(1300, 560, NB); lamp_post(1000, -400, NB)
group("bordur"); D(box(AX0, -608, NB, AX1, -600, NB + 24, "ho_gwall"))                   # güney kenar bordürü (atlanabilir)
sound((960, 0, NB + 200), "hamamonu/trafik.wav", 5, 8)
sound((700, 0, NB + 200), "hamamonu/kus.wav", 5, 4)
# A'nın doğusu: cephe evler (batıya bakar)
house_row("-x", 1408, -608, 640, NB, 100, floors=2, seed=5)
W(box(1504, 640, NB, 1536, 1024, NB + 200, "ho_gwall"))
# CT avlusu
for x, y in ((700, 1300), (1340, 1300)):
    tree(x, y, NB, 150, 90)
planter(900, 1340, 1120, 1400, NB)
lamp_post(640, 1100, NB); lamp_post(1400, 1100, NB)
W(box(700, 1420, NB, 1340, 1536, NB + 200, "ho_concrete"), box(512, 1420, NB, 700, 1536, NB + 120, "ho_gwall"), box(1340, 1420, NB, 1536, 1536, NB + 120, "ho_gwall"))
group("saksi"); D(box(560, 1060, NB, 624, 1100, NB + 40, "ho_ashlar"), box(1320, 1060, NB, 1384, 1100, NB + 40, "ho_ashlar"))
sound((1000, 1200, NB + 200), "hamamonu/trafik.wav", 6, 8)

# ================================================================ OYUN VARLIKLARI
n = 0
for y in (-1080, -1020, -960, -900):
    for x in (-1300, -1230, -1160, -1090):
        ent("info_player_deathmatch", (x, y, 40), angles="0 90 0"); n += 1
for y in (1120, 1180, 1240, 1300):
    for x in (780, 860, 940, 1020):
        ent("info_player_start", (x, y, NB + 40), angles="0 270 0")
ent("func_bomb_target", brushes=[box(-1200, -200, 0, -840, 200, 200, "aaatrigger")])          # B: saat kulesi
ent("func_bomb_target", brushes=[box(760, -300, NB, 1200, 140, NB + 200, "aaatrigger")])      # A: Hacettepe önü (anıt)
ent("info_map_parameters", (0, 0, 600), bombradius=500)

# ================================================================ IŞIK ve SESLER
ent("light_environment", (0, 0, 900), angles="0 60 0", pitch=-55,
    _light="255 244 222 200", _diffuse_light="150 180 230 110")
sound((0, -700, 100), "hamamonu/ruzgar.wav", 4, 8)
sound((-576, -200, 100), "hamamonu/ruzgar.wav", 3, 4)
sound((1100, -200, NB + 100), "hamamonu/korna.wav", 8, 4 + 16 + 32, targetname="ses_korna")
sound((-1024, 0, 300), "hamamonu/guvercin.wav", 7, 4 + 16 + 32, targetname="ses_guvercin")
sound((-100, -400, 100), "hamamonu/kopek.wav", 7, 4 + 16 + 32, targetname="ses_kopek")
sound((-1088, -960, 80), "hamamonu/guvercin.wav", 6, 4 + 16 + 32, targetname="ses_guvercin2")
ents.append(({"classname": "multi_manager", "targetname": "mm_sesler", "spawnflags": "1",
              "ses_guvercin": "7", "ses_korna": "19", "ses_kopek": "31", "ses_guvercin2": "40", "mm_sesler": "53"}, []))
ent("trigger_auto", (0, 0, 64), target="mm_sesler", spawnflags=1)

# ================================================================ YAZ
W(GROUND, UPPER)
for name, bs, shadow in DETAIL:
    if bs:
        props = {"zhlt_lightflags": 2} if shadow else {}
        ent("func_wall", brushes=bs, **props)


def write(path):
    out = ['{', '"classname" "worldspawn"', '"mapversion" "220"', f'"wad" "{WADS}"',
           '"skyname" "hamamonu"', '"MaxRange" "8192"', '"message" "Hamamonu, Ankara"', '"sounds" "1"']
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
    write(os.path.join(HERE, "de_hamamonu.map"))
