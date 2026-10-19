"""Tiny software renderer for GoldSrc BSP v30 (textures + lightmaps) for previews."""
import struct, sys, math
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


def render(path, eye, target, out, W=1280, H=720, fov=75):
    faces, texinfo, texs, light = load(path)
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
        if lo >= 0 and style != 255:
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
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)


if __name__ == "__main__":
    bsp = sys.argv[1]
    views = {
        "sea_view": ((-2300, -900, 420), (-500, 150, 380)),
        "aerial": ((-2400, 1500, 1350), (300, -200, 0)),
        "hall": ((-580, -420, 70), (-330, 300, 80)),
        "platforms": ((1450, -330, 72), (0, 100, 120)),
        "b_site": ((-660, -830, 110), (-1250, -1040, 30)),
        "courtyard": ((480, 0, 72), (-256, 0, 200)),
        "plan": ((-352, -1, 12000), (-352, 0, 0)),
    }
    which = sys.argv[2:] or list(views)
    for k in which:
        e, t = views[k]
        if k == "plan":
            render(bsp, e, t, f"/home/claude/hp/prev_{k}.png", W=1600, H=1200, fov=21)
        else:
            render(bsp, e, t, f"/home/claude/hp/prev_{k}.png")
        print("rendered", k)
