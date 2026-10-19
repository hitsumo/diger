"""Generates de_haydarpasa.map (Valve 220 format) for Counter-Strike 1.6.

Axes: +x = east (toward the tracks / Anatolia), -x = west (sea, Bosphorus),
+y = north. Ground level z = 0, sea surface z = -32.
"""
import itertools
import math
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WAD_PATH = os.path.join(HERE, "haydarpasa.wad")


def wad_sizes(path):
    """Texture name -> (w, h) from a WAD3 directory."""
    d = open(path, "rb").read()
    magic, n, off = struct.unpack_from("<4sii", d, 0)
    assert magic == b"WAD3", path
    sizes = {}
    for i in range(n):
        fo, _, _, _, _, _, name = struct.unpack_from("<iiiBBH16s", d, off + 32 * i)
        name = name.split(b"\0")[0].decode()
        sizes[name] = struct.unpack_from("<II", d, fo + 16)
    return sizes


TEXSIZE = wad_sizes(WAD_PATH)
TEXSIZE.update({"sky": (16, 16), "aaatrigger": (16, 16), "clip": (16, 16), "null": (16, 16)})
# yazılı vagon dokuları 512x128 ama hp_carside'ın (128) kaydırmasıyla hizalanır (yazi_yama.py)
TEXSIZE.update({"hp_carsc_guney": (128, 128), "hp_carsc_kuzey": (128, 128)})

# derleme: sdHLCSG -wadinclude haydarpasa  (sdhlt.wad yalnız sky/aaatrigger araç dokuları için)
WADS = "haydarpasa.wad;sdhlt.wad"


# ---------------------------------------------------------------- brushes
class Brush:
    def __init__(self, verts, tex, align="world", scale=(1.0, 1.0), fit=None):
        self.verts = [tuple(int(round(c)) for c in v) for v in verts]
        self.tex = tex
        self.align = align
        self.scale = scale
        self.fit = fit  # None | "v" | "uv" | "u"

    def faces(self):
        P = np.array(self.verts, float)
        c = P.mean(axis=0)
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
            # hlcsg normal = (p0 - p1) x (p2 - p1)
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
        return t.get("side", t.get("all", "hp_stone"))

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
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    return Brush(v, tex, **kw)


def prism(poly, z0, z1, tex, **kw):
    return Brush([(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly], tex, **kw)


def ngon(cx, cy, r, n, rot=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n + rot), cy + r * math.sin(2 * math.pi * i / n + rot))
            for i in range(n)]


def pyramid(poly, z0, apex, tex, **kw):
    return Brush([(x, y, z0) for x, y in poly] + [apex], tex, **kw)


def hip_roof(x0, y0, x1, y1, z0, z1, tex, **kw):
    """Hip roof over rectangle, ridge along the longer axis."""
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


def wall_x(x0, x1, y0, y1, z0, z1, openings, tex, **kw):
    """Wall running along y (thickness in x), with door openings (oy0, oy1, oz1)."""
    out, cur = [], y0
    for oy0, oy1, oz1 in sorted(openings):
        if oy0 > cur:
            out.append(box(x0, cur, z0, x1, oy0, z1, tex, **kw))
        if oz1 < z1:
            out.append(box(x0, oy0, oz1, x1, oy1, z1, tex, **kw))
        cur = oy1
    if cur < y1:
        out.append(box(x0, cur, z0, x1, y1, z1, tex, **kw))
    return out


def wall_y(y0, y1, x0, x1, z0, z1, openings, tex, **kw):
    out, cur = [], x0
    for ox0, ox1, oz1 in sorted(openings):
        if ox0 > cur:
            out.append(box(cur, y0, z0, ox0, y1, z1, tex, **kw))
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
    """Subtract the axis-aligned box `hole` = (x0, y0, z0, x1, y1, z1) from a list of box brushes."""
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
        e["origin"] = "%d %d %d" % origin
    e.update({k: str(v) for k, v in props.items()})
    ents.append((e, brushes or []))


def W(*bs):
    for b in bs:
        if isinstance(b, list):
            world.extend(b)
        else:
            world.append(b)


# ================================================================ BUILD
# ---- sky box (inner volume x -2528..1824, y -1632..1632, z -288..1472)
X0, X1, Y0, Y1, Z0, Z1 = -2528, 1824, -1632, 1632, -288, 1472
T = 32
W(box(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T, "sky"),
  box(X0 - T, Y0 - T, Z0 - T, X1 + T, Y1 + T, Z0, "hp_sand"),
  box(X0 - T, Y0, Z0, X0, Y1, Z1, "sky"),
  box(X1, Y0, Z0, X1 + T, Y1, Z1, "sky"),
  box(X0 - T, Y0 - T, Z0, X1 + T, Y0, Z1, "sky"),
  box(X0 - T, Y1, Z0, X1 + T, Y1 + T, Z1, "sky"))

# ---- ground (zemin plakaları: yer altı geçidi bunlardan oyulduktan sonra yazılır, bkz. UNDERPASS)
GROUND = [box(-1024, Y0, Z0, -640, Y1, 0, {"top": "hp_concrete", "side": "hp_rustic"}),   # rıhtım (quay)
          box(-640, Y0, Z0, 256, Y1, 0, {"top": "hp_cobble", "side": "hp_rustic"}),
          box(256, Y0, Z0, X1, Y1, 0, {"top": "hp_gravel", "side": "hp_rustic"})]
W(box(X0, Y0, Z0, -1024, Y1, -256, "hp_sand"))                                    # seabed

# quay edge kerb stones (visual band on top edge of quay)
W(box(-1024, -1152, 0, -1008, 1152, 4, "hp_cornice"))

# ---- boundary walls (brick warehouses / harbour walls)
BR = {"side": "hp_brick", "top": "hp_cornice", "bottom": "hp_brick"}
W(box(-1024, -1184, 0, X1, -1152, 320, BR),
  box(-1024, 1152, 0, X1, 1184, 320, BR),
  box(1792, -1152, 0, X1, 1152, 320, BR))
# warehouse roofs behind the walls (silhouette)
W(box(-640, -1632, 0, X1, -1184, 288, BR), box(-640, 1184, 0, X1, 1632, 288, BR),
  hip_roof(-656, -1632, X1, -1168, 288, 400, "hp_roof"),
  hip_roof(-656, 1168, X1, 1632, 288, 400, "hp_roof"))

# ================================================================ STATION
EXT_LOW = "hp_rustic"
EXT_UP = "hp_window"
IN_LOW = "hp_inwall"
IN_UP = "hp_plaster"

# --- main block: x -640..-256, y -640..640 ; walls to z 512 ; hall x -608..-288, y -480..480, z 0..256
west_doors = [(-136, -56, 128), (-40, 40, 128), (56, 136, 128)]
east_doors = [(-352, -256, 128), (-48, 48, 128), (256, 352, 128)]
W(wall_x(-640, -608, -640, 640, 0, 128, west_doors,
         {"-x": EXT_LOW, "+x": IN_LOW, "side": "hp_stone", "tb": "hp_stone"}))
W(box(-640, -640, 128, -608, 640, 512, {"-x": EXT_UP, "+x": IN_UP, "side": "hp_stone", "tb": "hp_stone"}))
W(wall_x(-288, -256, -640, 640, 0, 128, east_doors,
         {"+x": EXT_LOW, "-x": IN_LOW, "side": "hp_stone", "tb": "hp_stone"}))
W(box(-288, -640, 128, -256, 640, 512, {"+x": EXT_UP, "-x": IN_UP, "side": "hp_stone", "tb": "hp_stone"}))
# end blocks (solid ends of the building) north/south of the hall
for s in (-1, 1):
    ya, yb = sorted((s * 480, s * 640))
    inner = "-y" if s > 0 else "+y"
    outer = "+y" if s > 0 else "-y"
    W(box(-608, ya, 0, -288, yb, 128, {inner: IN_LOW, outer: EXT_LOW, "side": "hp_stone", "tb": "hp_stone"}),
      box(-608, ya, 128, -288, yb, 512, {inner: IN_UP, outer: EXT_UP, "side": "hp_stone", "tb": "hp_stone"}))
# hall floor, ceiling + upper floors fill
W(box(-608, -480, 0, -288, 480, 2, {"top": "hp_floor", "side": "hp_floor", "bottom": "hp_floor"}),
  box(-608, -480, 256, -288, 480, 512, {"bottom": "hp_ceil", "side": IN_UP, "top": "hp_stone"}))
# door thresholds inside openings (marble)
for oy0, oy1, _ in west_doors:
    W(box(-640, oy0, 0, -608, oy1, 2, {"top": "hp_floor", "side": "hp_stone"}))
for oy0, oy1, _ in east_doors:
    W(box(-288, oy0, 0, -256, oy1, 2, {"top": "hp_floor", "side": "hp_stone"}))

# string course + main cornice + hip roof
W(box(-648, -648, 128, -640, 648, 140, "hp_cornice"),     # string course: outer strips only
  box(-256, -648, 128, -248, 648, 140, "hp_cornice"),
  box(-640, -648, 128, -256, -640, 140, "hp_cornice"),
  box(-640, 640, 128, -256, 648, 140, "hp_cornice"),
  box(-656, -656, 512, -240, 656, 528, "hp_cornice"),
  hip_roof(-656, -656, -240, 656, 528, 704, {"slope": "hp_roof", "side": "hp_roof", "bottom": "hp_cornice"}))

# pilasters on the sea facade (between window bays)
for yc in (-512, -384, -256, 256, 384, 512):
    W(box(-656, yc - 12, 140, -640, yc + 12, 512, "hp_stone"))
    W(box(-256, yc - 12, 140, -240, yc + 12, 512, "hp_stone"))

# central risalit on the sea facade with pediment, sign and grand window
W(wall_x(-672, -640, -224, 224, 0, 128, west_doors, {"-x": EXT_LOW, "side": "hp_stone", "tb": "hp_stone"}))
W(box(-672, -224, 128, -640, 224, 576, "hp_stone"),
  box(-680, -232, 576, -632, 232, 592, "hp_cornice"),
  Brush([(-680, -232, 592), (-680, 232, 592), (-560, -232, 592), (-560, 232, 592),
         (-680, 0, 688), (-560, 0, 688)], {"side": "hp_stone", "slope": "hp_roof", "top": "hp_roof"}),
  box(-676, -128, 168, -672, 128, 424, "hp_window", align="local", fit="uv"),
  box(-676, -192, 456, -672, 192, 520, "hp_sign", align="local", fit="uv"))
# door reveals in risalit share the openings of the wall behind it
# small risalit on courtyard side with clock-like sign repeat
W(box(-256, -160, 128, -240, 160, 544, "hp_stone"),
  box(-244, -128, 448, -240, 128, 512, "hp_sign", align="local", fit="uv"),
  box(-248, -168, 544, -232, 168, 560, "hp_cornice"))

# --- the two round towers at the sea-facing corners
for s in (-1, 1):
    cx, cy = -640, s * 640
    W(prism(ngon(cx, cy, 144, 16), 0, 128, "hp_rustic", align="local"),
      prism(ngon(cx, cy, 152, 16), 128, 140, "hp_cornice"),
      prism(ngon(cx, cy, 144, 16), 140, 640, {"side": "hp_towerwin", "tb": "hp_stone"}, fit="u"),
      prism(ngon(cx, cy, 160, 16), 640, 656, "hp_cornice"),
      pyramid(ngon(cx, cy, 160, 16), 656, (cx, cy, 912), {"side": "hp_roof", "slope": "hp_roof", "bottom": "hp_cornice"}),
      prism(ngon(cx, cy, 8, 8), 912, 960, "hp_steel"))      # finial

# --- side wings flanking the platforms: x -256..256, |y| 416..640
for s in (-1, 1):
    ya, yb = sorted((s * 416, s * 640))
    W(box(-256, ya, 0, 256, yb, 128, {"side": EXT_LOW, "tb": "hp_stone"}),
      box(-256, ya, 128, 256, yb, 384, {"side": EXT_UP, "tb": "hp_stone"}),
      box(-264, ya - 8, 128, 264, yb + 8, 140, "hp_cornice"),
      box(-272, ya - 16, 384, 272, yb + 16, 400, "hp_cornice"),
      hip_roof(-272, ya - 16, 272, yb + 16, 400, 528, {"slope": "hp_roof", "side": "hp_roof", "bottom": "hp_cornice"}))
    # fake doors on the courtyard side and on the outer road side
    yi = s * 416
    yo = s * 640
    for xd in (-160, 32):
        W(box(xd, yi - s * 4, 0, xd + 64, yi, 128, "hp_door", align="local", fit="uv"),
          box(xd + 64, yo, 0, xd, yo + s * 4, 128, "hp_door", align="local", fit="uv"))
    # door at the track-end of the wing
    W(box(256, s * 496 - 32, 0, 260, s * 496 + 32, 128, "hp_door", align="local", fit="uv"))

# ================================================================ HALL FURNITURE
for yc in (-272, 272):
    for xc in (-448,):
        W(box(xc - 16, yc - 16, 2, xc + 16, yc + 16, 256, {"side": "hp_stone", "tb": "hp_stone"}))
# ticket counters (gişe) at both ends
for s in (-1, 1):
    ya, yb = sorted((s * 432, s * 464))
    W(box(-576, ya, 2, -352, yb, 48, {"side": "hp_wood", "top": "hp_wood"}))
# benches
for yb in (-160, 96):
    W(box(-400, yb, 2, -336, yb + 64, 22, "hp_wood"))

# ================================================================ COURTYARD / PLATFORMS
TRACKS = [(-832, 288), (-512, 288), (-160, -128), (160, -128), (512, 288), (832, 288)]
TRACK_END = 1536
for yc, xs in TRACKS:
    W(box(xs, yc - 56, 0, TRACK_END, yc + 56, 4, {"top": "hp_ballast", "side": "hp_gravel", "bottom": "hp_gravel"}))
    for off in (-28, 24):
        W(box(xs, yc + off, 4, TRACK_END, yc + off + 4, 12, "hp_rail"))
    W(box(xs - 32, yc - 48, 0, xs, yc + 48, 48, "hp_bumper"),
      box(TRACK_END, yc - 48, 0, TRACK_END + 32, yc + 48, 48, "hp_bumper"))

PLAT = {"top": "hp_plat", "side": "hp_platedge", "bottom": "hp_plat"}
PLATFORMS = [(-104, 104, -96), (-440, -232, 288), (232, 440, 288), (-760, -584, 288), (584, 760, 288)]
PLAT_BRUSHES = [box(xs, y0, 0, 1408, y1, 16, PLAT, align="world") for y0, y1, xs in PLATFORMS]

# platform canopies (peron sundurmaları)
for y0, y1, xs in PLATFORMS:
    yc = (y0 + y1) // 2
    for xp in (384, 640, 896, 1152):
        if yc == 0 and xp == 1152:
            xp = 1120          # orta peron: direk merdiven boşluğunun (x 1152..1408) kenarından geri çekildi
        W(box(xp - 8, yc - 8, 16, xp + 8, yc + 8, 224, "hp_steel"))
    W(box(320, y0 - 24, 224, 1216, y1 + 24, 236, {"bottom": "hp_canopy", "top": "hp_roof", "side": "hp_steel"}))
    W(box(320, yc - 4, 236, 1216, yc + 4, 248, "hp_steel"))

# kiosk on the central platform (A site cover)
W(box(560, -48, 16, 656, 48, 112, {"side": "hp_boxcar", "top": "hp_carroof", "bottom": "hp_carroof"}),
  box(552, -56, 112, 664, 56, 120, "hp_steel"))
# luggage crates on platforms near A
W(box(400, -400, 16, 464, -336, 80, "hp_crate", align="local"),
  box(704, 280, 16, 768, 344, 80, "hp_crate", align="local"),
  box(704, 344, 16, 752, 392, 64, "hp_crate", align="local"),
  box(96, -360, 0, 160, -296, 64, "hp_crate", align="local"))


# ---- rolling stock
def coach(yc, x0, x1, tex_n="hp_carside", tex_s="hp_carside"):
    W(box(x0, yc - 48, 16, x1, yc + 48, 152,
          {"+y": tex_n, "-y": tex_s, "side": "hp_carroof", "tb": "hp_carroof"}, fit="v"),
      Brush([(x0, yc - 48, 152), (x0, yc + 48, 152), (x1, yc - 48, 152), (x1, yc + 48, 152),
             (x0, yc - 36, 168), (x0, yc + 36, 168), (x1, yc - 36, 168), (x1, yc + 36, 168)], "hp_carroof"))


def loco(yc, x0, x1):
    W(box(x0, yc - 48, 16, x1, yc + 48, 176,
          {"+y": "hp_loco", "-y": "hp_loco", "side": "hp_locofront", "tb": "hp_carroof"}, fit="v"),
      Brush([(x0, yc - 48, 176), (x0, yc + 48, 176), (x1, yc - 48, 176), (x1, yc + 48, 176),
             (x0, yc - 32, 192), (x0, yc + 32, 192), (x1, yc - 32, 192), (x1, yc + 32, 192)], "hp_carroof"))


def boxcar(yc, x0, x1):
    W(box(x0, yc - 48, 16, x1, yc + 48, 160,
          {"+y": "hp_boxcar", "-y": "hp_boxcar", "side": "hp_boxcar", "tb": "hp_carroof"}, fit="v"),
      box(x0 - 4, yc - 52, 160, x1 + 4, yc + 52, 168, "hp_carroof"))


coach(-160, 896, 1408)
coach(160, 544, 1024, "hp_carsc_kuzey", "hp_carsc_guney")    # "sicakcikolata.com" yazılı vagon
coach(160, 1056, 1536)
loco(-512, 640, 960)
coach(-512, 992, 1472)
boxcar(512, 352, 704)
boxcar(512, 736, 1088)
boxcar(832, 800, 1152)

# ================================================================ QUAY, PIER, FERRY
# bollards along the quay edge
for yb in range(-1088, 1153, 224):
    W(prism(ngon(-992, yb, 12, 8), 0, 28, "hp_steel"))
# lamp posts
for yl in (-896, -384, 160, 896):      # 384 -> 160: geçit çıkışı yerine
    W(box(-916, yl - 4, 0, -908, yl + 4, 176, "hp_steel"), box(-924, yl - 12, 176, -900, yl + 12, 184, "hp_steel"))
    ent("light", (-912, yl, 164), _light="255 214 160 90")

# cover on the quay
W(box(-976, -720, 0, -848, -400, 128, {"side": "hp_contblue", "tb": "hp_carroof"}),
  box(-944, -1120, 0, -880, -1056, 64, "hp_crate", align="local"),
  box(-944, -1120, 64, -880, -1056, 128, "hp_crate", align="local"),
  box(-816, -880, 0, -752, -816, 64, "hp_crate", align="local"),
  box(-816, 1000, 0, -752, 1064, 64, "hp_crate", align="local"),    # (-880,432) -> geçit çıkışının kuzeyine
  box(-976, 560, 0, -848, 880, 128, {"side": "hp_contred", "tb": "hp_carroof"}))

# pier (iskele) with pier house
W(box(-1600, -128, -16, -1024, 128, 0, {"top": "hp_wood", "side": "hp_wood", "bottom": "hp_wood"}))
for xp in range(-1584, -1024, 128):
    for yp in (-112, 112):
        W(box(xp - 8, yp - 8, -256, xp + 8, yp + 8, -16, "hp_wood"))
PH = {"side": "hp_plaster", "tb": "hp_stone"}
W(box(-1600, -160, -16, -1408, 160, 0, {"top": "hp_wood", "side": "hp_wood", "bottom": "hp_wood"}))
W(wall_x(-1600, -1584, -160, 160, 0, 176, [(-64, 64, 128)], PH),
  wall_x(-1424, -1408, -160, 160, 0, 176, [(-64, 64, 128)], PH),
  box(-1584, -160, 0, -1424, -144, 176, {"-y": "hp_window", "side": "hp_plaster", "tb": "hp_stone"}, align="local"),
  box(-1584, 144, 0, -1424, 160, 176, {"+y": "hp_window", "-y": "hp_logo", "side": "hp_plaster", "tb": "hp_stone"}, align="local"),
  box(-1600, -160, 176, -1408, 160, 192, "hp_cornice"),
  hip_roof(-1640, -200, -1368, 200, 192, 304, {"slope": "hp_roof", "side": "hp_roof", "bottom": "hp_canopy"}))
ent("light", (-1504, 0, 150), _light="255 214 160 120")

# sea
W(box(X0, Y0, -256, -1024, Y1, -32, "!hp_sea"))

# vapur (ferry) moored at the south quay
hull = [(-1312, -1280), (-1088, -1280), (-1088, -864), (-1120, -800), (-1280, -800), (-1312, -864)]
bow = [(-1312, -1280), (-1200, -1392), (-1088, -1280)]
for poly in (hull, bow):
    W(prism(poly, -96, -8, "hp_hullb"),
      prism(poly, -8, 32, {"side": "hp_hullw", "top": "hp_wood", "bottom": "hp_hullb"}))
W(box(-1264, -1216, 32, -1136, -896, 128, {"side": "hp_cabin", "tb": "hp_carroof"}, fit="v"),
  box(-1280, -1232, 128, -1120, -880, 136, "hp_carroof"),
  prism(ngon(-1200, -1056, 28, 8), 136, 240, {"side": "hp_funnel", "tb": "hp_steel"}, fit="uv"),
  box(-1312, -1280, 32, -1304, -864, 56, "hp_steel"),
  box(-1304, -808, 32, -1128, -800, 56, "hp_steel"))
# gangplanks (iskele köprüsü) quay -> deck
for y0 in (-1088, -960):
    W(Brush([(-1088, y0, 32), (-1088, y0 + 64, 32), (-1024, y0, 0), (-1024, y0 + 64, 0),
             (-1088, y0, 24), (-1088, y0 + 64, 24), (-1024, y0, -8), (-1024, y0 + 64, -8)],
            {"top": "hp_wood", "slope": "hp_wood", "side": "hp_steel", "bottom": "hp_steel"}))

# ladders out of the sea
LADDERS = [(-1024, -560, "-x"), (-1024, 256, "-x"), (-1024, 960, "-x"), (-1216, -128, "-y")]
for lx, ly, face in LADDERS:
    if face == "-x":
        vis = box(lx - 4, ly, -256, lx, ly + 32, 0, "{hp_ladder", align="local")
        trig = box(lx - 20, ly, -256, lx, ly + 32, 16, "aaatrigger")
    else:
        vis = box(lx, ly - 4, -256, lx + 32, ly, 0, "{hp_ladder", align="local")
        trig = box(lx, ly - 20, -256, lx + 32, ly, 16, "aaatrigger")
    ent("func_illusionary", brushes=[vis], rendermode=4, renderamt=255)
    ent("func_ladder", brushes=[trig])

# ================================================================ ROADS / YARDS
# south road: crates + red container
W(box(96, -1120, 0, 416, -992, 128, {"side": "hp_contred", "tb": "hp_carroof"}),
  box(-200, -1000, 0, -136, -936, 64, "hp_crate", align="local"),
  box(-480, -1088, 0, -416, -1024, 64, "hp_crate", align="local"))
# north yard: containers
W(box(-400, 800, 0, -80, 928, 128, {"side": "hp_contblue", "tb": "hp_carroof"}),
  box(-368, 800, 128, -112, 928, 256, {"side": "hp_contred", "tb": "hp_carroof"}),
  box(64, 960, 0, 192, 1120, 128, {"side": "hp_contblue", "tb": "hp_carroof"}),
  box(-560, 1040, 0, -496, 1104, 64, "hp_crate", align="local"))

# ================================================================ UNDERPASS (yer altı geçidi)
# Orta peronun doğu ucundan (x 1152..1408) merdivenle inilir; peronların, avlunun ve bekleme
# salonunun altından geçip rıhtımda kuzey kulenin yanında (x -928..-800, y 224..480) merdivenle
# çıkar. Avlu altındaki dirsek uzun bir niş hattını kırar.
TUN_FLOOR, TUN_CEIL, LIN = -128, -24, 16
TUN_TEX = {"top": "hp_concrete", "bottom": "hp_concrete", "side": "hp_brick"}
SEGS = [(-96, -64, 1152, 64),        # peronların altı, doğu-batı (y 0 ekseni)
        (-224, -64, -96, 224),       # dirsek (avlunun altı)
        (-928, 96, -224, 224)]       # salon ve rıhtımın altı (y 160 ekseni)
PIT_E = (1152, -64, 1408, 64)        # doğu girişi: üst basamak peron ucunda, batıya iner
PIT_W = (-928, 224, -800, 480)       # batı çıkışı: kuzeye çıkar, y=480'de rıhtıma açılır
interiors = [(x0, y0, TUN_FLOOR, x1, y1, TUN_CEIL) for x0, y0, x1, y1 in SEGS]
interiors += [(x0, y0, TUN_FLOOR, x1, y1, 0) for x0, y0, x1, y1 in (PIT_E, PIT_W)]
shell = []
for x0, y0, z0, x1, y1, z1 in interiors:
    hole = (x0 - LIN, y0 - LIN, z0 - LIN, x1 + LIN, y1 + LIN, min(z1 + LIN, 0))
    GROUND = carve(GROUND, hole)
    shell.append(box(*hole, TUN_TEX))
for hole in interiors:
    shell = carve(shell, hole)
PLAT_BRUSHES = carve(PLAT_BRUSHES, (PIT_E[0], PIT_E[1], -1, PIT_E[2], PIT_E[3], 17))   # peronda merdiven boşluğu
W(GROUND, PLAT_BRUSHES, shell)
# basamaklar: 16 yüksek, 32 derin; en altta 32'lik sahanlık
for k in range(1, 8):
    W(box(1408 - 32 * k, -64, TUN_FLOOR, 1408 - 32 * (k - 1), 64, -16 * k, "hp_concrete"))      # doğu
    W(box(-928, 480 - 32 * k, TUN_FLOOR, -800, 480 - 32 * (k - 1), -16 * k, "hp_concrete"))     # batı
# rıhtım çıkışının çevresinde alçak korkuluk duvarı (üç yan; kuzey ucu açık)
W(box(-944, 208, 0, -928, 480, 24, "hp_stone"),
  box(-800, 208, 0, -784, 480, 24, "hp_stone"),
  box(-944, 208, 0, -784, 224, 24, "hp_stone"))
# tünel aydınlatması
for x in (1024, 768, 512, 256, 0):
    ent("light", (x, 0, TUN_CEIL - 16), _light="255 236 200 90")
for x, y in ((-160, 160), (-416, 160), (-672, 160), (-896, 160)):
    ent("light", (x, y, TUN_CEIL - 16), _light="255 236 200 90")

# ================================================================ GAMEPLAY ENTITIES
# CT spawns: inside the main waiting hall
for x in (-560, -496):
    for y in (-224, -160, -96, -32, 32, 96, 160, 224):
        ent("info_player_start", (x, y, 40), angles="0 0 0")
# T spawns: east end of the rail yard
n = 0
for x in (1632, 1696, 1760):
    for y in (-1088, -1024, -960, -896, -832, -768):
        if n < 16:
            ent("info_player_deathmatch", (x, y, 40), angles="0 180 0")
            n += 1

ent("func_bomb_target", brushes=[box(272, -472, 0, 832, 472, 256, "aaatrigger")])          # A: platforms
ent("func_bomb_target", brushes=[box(-1320, -1150, -40, -680, -760, 240, "aaatrigger")])   # B: ferry quay
ent("info_map_parameters", (0, 0, 600), bombradius=500)

# ================================================================ LIGHTS
ent("light_environment", (0, 0, 1200), angles="0 20 0", pitch=-40,
    _light="255 232 200 240", _diffuse_light="170 195 230 120")
for y in (-400, -240, -80, 80, 240, 400):
    for x in (-544, -352):
        ent("light", (x, y, 200), _light="255 214 160 140")
for y0, y1, xs in PLATFORMS:
    yc = (y0 + y1) // 2
    for xp in (512, 1024):
        ent("light", (xp, yc, 210), _light="255 236 200 110")
ent("light", (0, 0, 200), _light="255 230 190 120")
for y in (-280, 280):
    ent("light", (-128, y, 200), _light="255 230 190 150")

# ================================================================ WRITE
def write(path):
    out = ['{', '"classname" "worldspawn"', '"mapversion" "220"', f'"wad" "{WADS}"',
           '"skyname" "desert"', '"MaxRange" "8192"', '"message" "Haydarpasa Gari"', '"sounds" "1"']
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
    write(os.path.join(HERE, "de_haydarpasa.map"))
