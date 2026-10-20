"""de_tutun önizlemeleri: GoldSrc BSP v30 için küçük yazılım render'ı (dokular + ışık haritaları + gökyüzü)."""
import os, struct, sys, math
import numpy as np
from PIL import Image


def load(path):
    d = open(path, "rb").read()
    ver = struct.unpack_from("<i", d, 0)[0]
    assert ver == 30
    L = [struct.unpack_from("<ii", d, 4 + 8 * i) for i in range(15)]
    lump = lambda i: d[L[i][0]:L[i][0] + L[i][1]]
    verts = np.frombuffer(lump(3), "<f4").reshape(-1, 3)
    edges = np.frombuffer(lump(12), "<u2").reshape(-1, 2)
    surf = np.frombuffer(lump(13), "<i4")
    ti = lump(6)
    texinfo = [(np.frombuffer(ti[i:i + 32], "<f4").reshape(2, 4), struct.unpack_from("<i", ti, i + 32)[0])
               for i in range(0, len(ti), 40)]
    tl = lump(2)
    n = struct.unpack_from("<i", tl, 0)[0]
    offs = struct.unpack_from("<%di" % n, tl, 4)
    texs = []
    for o in offs:
        name = tl[o:o + 16].split(b"\0")[0].decode(errors="replace")
        w, h = struct.unpack_from("<II", tl, o + 16)
        m0 = struct.unpack_from("<I", tl, o + 24)[0]
        m3 = struct.unpack_from("<I", tl, o + 36)[0]
        if m0 == 0:
            texs.append((name, None)); continue
        idx = np.frombuffer(tl, np.uint8, w * h, o + m0).reshape(h, w)
        po = o + m3 + (w // 8) * (h // 8) + 2
        pal = np.frombuffer(tl, np.uint8, 768, po).reshape(256, 3)
        texs.append((name, pal[idx].astype(np.float32)))
    light = np.frombuffer(lump(8), np.uint8)
    fl = lump(7)
    faces, sides = [], []
    for i in range(0, len(fl), 20):
        pn, side, fe, ne, tix = struct.unpack_from("<HHiHH", fl, i)
        styles = fl[i + 12:i + 16]
        lo = struct.unpack_from("<i", fl, i + 16)[0]
        ids = []
        for k in range(fe, fe + ne):
            e = surf[k]
            ids.append(edges[e][0] if e >= 0 else edges[-e][1])
        faces.append((verts[ids], tix, lo, styles[0]))
        sides.append((pn, side))
    models = lump(14)
    first, num = struct.unpack_from("<ii", models, 56)
    pl = lump(1)
    normals = [np.frombuffer(pl[i:i + 12], "<f4") for i in range(0, len(pl), 20)]
    faces = [(f[0], f[1], f[2], f[3], normals[pn] * (-1 if side else 1))
             for f, (pn, side) in zip(faces, sides)]
    return faces[first:first + num], texinfo, texs, light


# GoldSrc sky kutusu: yüz sırası rt(+x) bk(-x) lf(+y) ft(-y) up(+z) dn(-z); vec[j] = ±b[|k|-1], b=(s,t,1)
ST_TO_VEC = [(3, -1, 2), (-3, 1, 2), (1, 3, 2), (-1, -3, 2), (-2, -1, 3), (2, -1, -3)]
SKY_SUFFIX = ["rt", "bk", "lf", "ft", "up", "dn"]


def load_sky(bsp_path):
    """<harita>/../gfx/env/<skyname>*.tga varsa 6 yüzü (h, w, 3) float dizileri olarak döndürür."""
    d = open(bsp_path, "rb").read()
    o, n = struct.unpack_from("<ii", d, 4)
    ents = d[o:o + n].decode(errors="replace")
    import re
    m = re.search(r'"skyname"\s+"([^"]+)"', ents)
    if not m:
        return None
    env = os.path.join(os.path.dirname(os.path.abspath(bsp_path)), "..", "gfx", "env")
    faces = []
    for suf in SKY_SUFFIX:
        p = os.path.join(env, m.group(1) + suf + ".tga")
        if not os.path.exists(p):
            return None
        t = open(p, "rb").read()
        w, h, bpp = struct.unpack_from("<HHB", t, 12)
        desc = t[17]
        px = np.frombuffer(t, np.uint8, w * h * (bpp // 8), 18 + t[0]).reshape(h, w, bpp // 8)[:, :, :3][:, :, ::-1]
        if not desc & 0x20:
            px = px[::-1]
        faces.append(px.astype(np.float32))
    return faces


def sample_sky(sky, dirs):
    """dirs: (..., 3) -> (..., 3) RGB."""
    v = dirs.reshape(-1, 3)
    out = np.zeros_like(v)
    ax = np.argmax(np.abs(v), axis=1)
    sgn = np.sign(v[np.arange(len(v)), ax])
    face = ax * 2 + (sgn < 0)
    for i in range(6):
        sel = face == i
        if not sel.any():
            continue
        k = ST_TO_VEC[i]
        jn = [abs(q) for q in k].index(3); js = [abs(q) for q in k].index(1); jt = [abs(q) for q in k].index(2)
        vv = v[sel]
        dn = vv[:, jn] * (1 if k[jn] > 0 else -1)
        s = vv[:, js] * (1 if k[js] > 0 else -1) / dn
        t = vv[:, jt] * (1 if k[jt] > 0 else -1) / dn
        img = sky[i]
        h, w = img.shape[:2]
        x = np.clip((s + 1) / 2 * w - 0.5, 0, w - 1); y = np.clip((1 - t) / 2 * h - 0.5, 0, h - 1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
        out[sel] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
                    img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return out.reshape(dirs.shape)


def render(path, eye, target, out, W=1280, H=720, fov=75):
    faces, texinfo, texs, light = load(path)
    sky = load_sky(path)
    eye = np.array(eye, float); target = np.array(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fl = (W / 2) / math.tan(math.radians(fov) / 2)
    img = np.zeros((H, W, 3), np.float32)
    # sky gradient
    gy = np.linspace(0, 1, H)[:, None]
    img[:] = (np.array([110, 160, 215]) * (1 - gy) + np.array([200, 215, 230]) * gy)[:, None, :]
    zbuf = np.full((H, W), np.inf, np.float32)
    near = 4.0
    for poly, tix, lo, style, nrm in faces:
        if nrm @ (eye - poly[0]) <= 0:
            continue
        vecs, mt = texinfo[tix]
        name, tex = texs[mt]
        if tex is None or name.lower() in ("sky", "aaatrigger", "null", "clip"):
            continue
        th, tw = tex.shape[:2]
        st = poly @ vecs[:, :3].T + vecs[:, 3]
        lmins = np.floor(st.min(0) / 16); lmaxs = np.ceil(st.max(0) / 16)
        lw, lh = (lmaxs - lmins + 1).astype(int)
        if lo >= 0 and style != 255 and lo + lw * lh * 3 <= len(light):   # ışıksız (RAD öncesi) BSP de çizilsin
            lm = light[lo:lo + lw * lh * 3].reshape(lh, lw, 3).astype(np.float32)
        else:
            lm = None
        rel = poly - eye
        cam = np.stack([rel @ r, rel @ u, rel @ f], 1)
        data = np.concatenate([cam, st], 1)
        # clip against near plane
        outp = []
        nv = len(data)
        for i in range(nv):
            a, b = data[i], data[(i + 1) % nv]
            ain, bin_ = a[2] >= near, b[2] >= near
            if ain: outp.append(a)
            if ain != bin_:
                t = (near - a[2]) / (b[2] - a[2]); outp.append(a + (b - a) * t)
        if len(outp) < 3:
            continue
        P = np.array(outp)
        sx = W / 2 + P[:, 0] / P[:, 2] * fl
        sy = H / 2 - P[:, 1] / P[:, 2] * fl
        iz = 1 / P[:, 2]
        for k in range(1, len(P) - 1):
            idx = [0, k, k + 1]
            x = sx[idx]; y = sy[idx]
            x0 = max(int(np.floor(x.min())), 0); x1 = min(int(np.ceil(x.max())), W - 1)
            y0 = max(int(np.floor(y.min())), 0); y1 = min(int(np.ceil(y.max())), H - 1)
            if x0 > x1 or y0 > y1:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-9:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((y[1] - y[2]) * (X - x[2]) + (x[2] - x[1]) * (Y - y[2])) / den
            w1 = ((y[2] - y[0]) * (X - x[2]) + (x[0] - x[2]) * (Y - y[2])) / den
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not m.any():
                continue
            izp = w0 * iz[idx[0]] + w1 * iz[idx[1]] + w2 * iz[idx[2]]
            z = 1 / izp
            zb = zbuf[y0:y1 + 1, x0:x1 + 1]
            m &= z < zb
            if not m.any():
                continue
            S = (w0 * P[idx[0], 3] * iz[idx[0]] + w1 * P[idx[1], 3] * iz[idx[1]] + w2 * P[idx[2], 3] * iz[idx[2]]) * z
            T = (w0 * P[idx[0], 4] * iz[idx[0]] + w1 * P[idx[1], 4] * iz[idx[1]] + w2 * P[idx[2], 4] * iz[idx[2]]) * z
            S, T = S[m], T[m]
            col = tex[(np.floor(T).astype(int) % th), (np.floor(S).astype(int) % tw)]
            if name.startswith("{"):
                keep = ~np.all(col == [0, 0, 255], axis=1)
            else:
                keep = np.ones(len(col), bool)
            if lm is not None:
                ls = np.clip(S / 16 - lmins[0], 0, lw - 1); lt = np.clip(T / 16 - lmins[1], 0, lh - 1)
                a0 = np.floor(ls).astype(int); b0 = np.floor(lt).astype(int)
                a1 = np.minimum(a0 + 1, lw - 1); b1 = np.minimum(b0 + 1, lh - 1)
                fa = (ls - a0)[:, None]; fb = (lt - b0)[:, None]
                li = (lm[b0, a0] * (1 - fa) * (1 - fb) + lm[b0, a1] * fa * (1 - fb) +
                      lm[b1, a0] * (1 - fa) * fb + lm[b1, a1] * fa * fb)
                col = col * li / 160.0
            elif name.startswith("!"):
                col = np.broadcast_to(tex.reshape(-1, 3).mean(0) * 1.15, col.shape) + (col - tex.reshape(-1, 3).mean(0)) * 0.25
            ys, xs = np.nonzero(m)
            ys, xs, col, zz = ys[keep], xs[keep], col[keep], z[m][keep]
            img[y0 + ys, x0 + xs] = col
            zbuf[y0 + ys, x0 + xs] = zz
    if sky is not None:
        ys, xs = np.nonzero(np.isinf(zbuf))
        if len(ys):
            dirs = (f[None, :] + r[None, :] * ((xs + 0.5 - W / 2) / fl)[:, None] + u[None, :] * ((H / 2 - ys - 0.5) / fl)[:, None])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            img[ys, xs] = sample_sky(sky, dirs)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)


if __name__ == "__main__":
    # kullanım: python render_bsp.py harita.bsp [-o cikti_dizini] [gorunum ...]
    args = sys.argv[1:]
    bsp = args.pop(0)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "onizleme")
    if args[:1] == ["-o"]:
        args.pop(0); outdir = args.pop(0)
    views = {
        "plan": ((0, -1, 12000), (0, 0, 0)),
        "t_giris": ((240, -1440, 72), (0, -700, 140)),                # T girişinden kuzeye
        "avlu": ((40, -620, 80), (0, 0, 110)),                        # avlu ve makine
        "makine": ((300, 300, 140), (-40, -20, 110)),
        "fabrika": ((-690, 10, 72), (-1300, -420, 110)),              # fabrika, B tarafı
        "asmakat": ((-1420, 480, 230), (-1000, -500, 60)),            # asma kattan salon
        "depo_a": ((1260, -240, 72), (900, -520, 80)),               # A deposu içi
        "depo_kapi": ((560, -420, 72), (720, -400, 90)),             # sıkışmış kapı dıştan
        "cop": ((420, 900, 110), (820, 1200, 260)),                  # çöp sahası, baca, sarmal
        "yuruyus": ((800, 1010, 290), (800, 500, 220)),              # yürüyüş yolundan depo çatısına
        "ct": ((-1250, 1200, 80), (-900, 560, 110)),
        "sokak": ((1440, -700, 72), (1100, 0, 90)),                  # doğu sokağı
    }
    which = args or list(views)
    for k in which:
        e, t = views[k]
        out = os.path.join(outdir, f"prev_{k}.png")
        if k == "plan":
            render(bsp, e, t, out, W=1600, H=1200, fov=21)
        else:
            render(bsp, e, t, out)
        print("rendered", k)
