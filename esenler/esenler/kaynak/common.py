"""Esenler Otogarı haritaları için ortak fırça/varlık altyapısı (cs_esenler, de_esenler).

Valve 220 .map yazar. Doku boyutları yanındaki esenler.wad'dan okunur.
"""
import itertools
import math
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WAD_PATH = os.path.join(HERE, "esenler.wad")


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
WADS = "esenler.wad;sdhlt.wad"     # derleme: sdHLCSG -wadinclude esenler


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
        return t.get("side", t.get("all", "es_wall"))

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
            if fit and (us.max() - us.min() < 16 or vs.max() - vs.min() < 16):
                fit = None          # ince kenar yüzleri: sığdırma yok (surface extents sınırı 240 texel)
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
    """Duvar yüzeyine kabartma (tabela, pencere, afiş). face = yüzey normali yönü, c = yüzey koordinatı."""
    if face == "+x":
        return box(c, a0, z0, c + depth, a1, z1, tex, align="local", fit="uv")
    if face == "-x":
        return box(c - depth, a0, z0, c, a1, z1, tex, align="local", fit="uv")
    if face == "+y":
        return box(a0, c, z0, a1, c + depth, z1, tex, align="local", fit="uv")
    return box(a0, c - depth, z0, a1, c, z1, tex, align="local", fit="uv")


def stairs(axis, a0, a1, b0, b1, z0, z1, tex, step_h=16, step_d=None):
    """a0'da z0, a1'de z1 olan merdiven (iniş veya çıkış). Basamak kutuları alt kottan yükselir."""
    lo_z, hi_z = min(z0, z1), max(z0, z1)
    n = int(round((hi_z - lo_z) / step_h))
    d = step_d or abs(a1 - a0) / n
    lo_a = a0 if z0 < z1 else a1          # alçak uç
    s = 1 if (a1 if z0 < z1 else a0) > lo_a else -1
    out = []
    for k in range(1, n + 1):
        p0 = lo_a + s * d * (k - 1)
        p1 = lo_a + s * d * k
        top = lo_z + step_h * k
        if axis == "x":
            out.append(box(min(p0, p1), b0, lo_z, max(p0, p1), b1, top, tex))
        else:
            out.append(box(b0, min(p0, p1), lo_z, b1, max(p0, p1), top, tex))
    return out


def ramp(axis, a0, a1, b0, b1, z0, z1, tex):
    """a0'da z0, a1'de z1 olan dolu rampa."""
    base = min(z0, z1)
    if axis == "x":
        pts = [(a0, b0, base), (a0, b1, base), (a1, b0, base), (a1, b1, base), (a0, b0, z0), (a0, b1, z0), (a1, b0, z1), (a1, b1, z1)]
    else:
        pts = [(b0, a0, base), (b1, a0, base), (b0, a1, base), (b1, a1, base), (b0, a0, z0), (b1, a0, z0), (b0, a1, z1), (b1, a1, z1)]
    return Brush(pts, tex)


# ---------------------------------------------------------------- figürler (otogar insanları)
FIG_MODEL = "models/hostage.mdl"       # cstrike'ın kendi sivil modeli (4 kıyafet: skin 0..3)
_fig_n = [0]


def figure(x, y, z, yaw, skin=0, blink=None):
    """cycler_sprite ile sivil figür (otogarın kendi modeli, 4 kıyafet). blink=(gorunme_sn, bosluk_sn) verilirse
    başta görünmez; multi_manager zinciri figürleri sırayla gösterip gizler (ara ara görünürler)."""
    _fig_n[0] += 1
    name = "fig%d" % _fig_n[0]
    props = dict(model=FIG_MODEL, framerate=8, skin=skin, angles="0 %d 0" % yaw, rendermode=2,
                 renderamt=(0 if blink else 255), targetname=name)
    ent("cycler_sprite", (x, y, z), **props)
    if blink:
        BLINKS.append((name, blink))
    return name


BLINKS = []


def finish_blinks():
    if not BLINKS:
        return
    for name, _ in BLINKS:
        ent("env_render", (0, 0, 0), targetname=name + "_goster", target=name, rendermode=2, renderamt=255, spawnflags=9)
        ent("env_render", (0, 0, 0), targetname=name + "_gizle", target=name, rendermode=2, renderamt=0, spawnflags=9)
    chunks = [BLINKS[i:i + 7] for i in range(0, len(BLINKS), 7)]       # multi_manager en çok 16 hedef
    for ci, chunk in enumerate(chunks):
        mm = {"classname": "multi_manager", "targetname": "mm_fig%d" % ci, "spawnflags": "1"}
        t = 2.0
        for name, (on_s, off_s) in chunk:
            mm[name + "_goster"] = "%.1f" % t
            mm[name + "_gizle"] = "%.1f" % (t + on_s)
            t += on_s + off_s
        nxt = "mm_fig%d" % ((ci + 1) % len(chunks))
        mm[nxt] = "%.1f" % (t + 1.0)
        ents.append((mm, []))
    ent("trigger_auto", (0, 0, 0), target="mm_fig0", spawnflags=1)


def light(origin, rgb, brightness, style=0, **kw):
    props = {"_light": "%d %d %d %d" % (*rgb, brightness)}
    if style:
        props["style"] = style
    props.update(kw)
    ent("light", origin, **props)


def sound(origin, wav, volume=7, flags=4, **kw):
    """ambient_generic: flags 1=her yerde 2=küçük 4=orta 8=büyük yarıçap 16=sessiz başla 32=tekrarsız."""
    ent("ambient_generic", origin, message=wav, health=volume, spawnflags=flags, pitch=100, **kw)


def fluorescent(x0, y0, x1, y1, z, rgb=(235, 240, 255), brightness=120, style=0, flicker=False):
    """Tavana gömülü floresan: emissif doku şeridi + nokta ışık."""
    group("floresan")
    D(box(x0, y0, z - 2, x1, y1, z, {"bottom": "es_light", "side": "es_metal", "top": "es_metal"}, align="local", fit="uv"))
    light(((x0 + x1) / 2, (y0 + y1) / 2, z - 12), rgb, brightness, style=(10 if flicker else style))


def write(path, skyname, message, maxrange=8192):
    for name, bs, shadow in DETAIL:
        if bs:
            props = {"zhlt_lightflags": 2} if shadow else {}
            ent("func_wall", brushes=bs, **props)
    finish_blinks()
    out = ['{', '"classname" "worldspawn"', '"mapversion" "220"', f'"wad" "{WADS}"',
           f'"skyname" "{skyname}"', f'"MaxRange" "{maxrange}"', f'"message" "{message}"', '"sounds" "1"']
    out += [b.to_map() for b in world]
    out.append("}")
    for e, bs in ents:
        out.append("{")
        out += [f'"{k}" "{v}"' for k, v in e.items()]
        out += [b.to_map() for b in bs]
        out.append("}")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"{os.path.basename(path)}: {len(world)} world brushes, {len(ents)} entities")
