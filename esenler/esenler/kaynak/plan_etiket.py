"""prev_plan.png üstüne bölge etiketleri -> onizleme/plan_<harita>.png
Kullanım: python plan_etiket.py cs_esenler|de_esenler [girdi] [cikti]
"""
import os, sys, math
from PIL import Image, ImageDraw, ImageFont

here = os.path.dirname(os.path.abspath(__file__))
which = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "onizleme", "%s_plan.png" % which)
dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "..", "onizleme", "plan_%s.png" % which)
im = Image.open(src).convert("RGB")
W, H = im.size
CX, CY, DIST = 0, 0, 14000
half = math.tan(math.radians(10.5)) * DIST
sx = W / 2 / half
P = lambda x, y: (W / 2 + (x - CX) * sx, H / 2 - (y - CY) * sx)
d = ImageDraw.Draw(im, "RGBA")
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
def label(x, y, text, font=f, fill=(255, 255, 255)):
    bb = d.textbbox((0, 0), text, font=font)
    px, py = P(x, y)
    px -= (bb[2] - bb[0]) / 2; py -= (bb[3] - bb[1]) / 2
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        d.text((px + ox, py + oy), text, font=font, fill=(0, 0, 0))
    d.text((px, py), text, font=font, fill=fill)
def zone(x0, y0, x1, y1, rgb):
    d.rectangle([P(x0, y1), P(x1, y0)], fill=rgb + (70,), outline=rgb + (255,), width=3)
if which == "cs_esenler":
    zone(-224, -1600, 224, -1344, (60, 130, 240)); label(0, -1470, "CT", f2, (120, 170, 255))
    zone(1216, -800, 1408, 800, (60, 130, 240)); label(1312, 0, "CT (metro)", f, (120, 170, 255))
    zone(-400, -120, 100, 120, (230, 170, 40)); label(-150, 0, "T", f2, (255, 200, 60))
    for x, y, t in ((-320, -300, "rehine"), (-400, 300, "rehine"), (300, 340, "rehine (1. kat)"), (-380, -300, "rehine (1. kat)")):
        pass
    label(-380, -300, "rehineler: salon, PTT, 1. kat", fs)
    label(0, 560, "ORTA BİNA (2 kat)"); label(200, -200, "yürüyen merdiven → metro", fs)
    label(688, 120, "bilet holü (-288)", fs); label(1072, 120, "merdiven", fs); label(1312, 900, "metro peronu (-480)", fs)
    label(648, 1100, "kuzey metro çıkışı", fs)
    label(0, -1000, "güney peronlar (12-15)"); label(0, 1000, "kuzey peronlar (16-19)")
    label(-1400, 0, "batı peronlar", fs); label(1500, -700, "doğu peronlar", fs)
    label(0, -1270, "yazıhaneler", fs); label(0, 1270, "yazıhaneler", fs)
    label(-1150, -1100, "otobüsler", fs); label(1150, 1100, "otobüsler", fs)
else:
    zone(-1560, -1340, -1180, -1020, (230, 170, 40)); label(-1370, -1180, "T", f2, (255, 200, 60))
    zone(-180, 920, 180, 1080, (60, 130, 240)); label(0, 1000, "CT", f2, (120, 170, 255))
    zone(1200, -512, 1600, -80, (230, 60, 40)); label(1400, -300, "A", f2, (255, 90, 70))
    zone(-1600, 80, -1056, 512, (240, 200, 40)); label(-1330, 300, "B", f2, (255, 220, 60))
    label(1400, -560, "tuvalet", fs); label(-1330, 560, "depo", fs)
    label(-1200, -700, "rampa", fs); label(-1330, -200, "yükleme rıhtımı", fs)
    label(0, -432, "güney koridor", fs); label(0, 0, "orta koridor", fs); label(0, 432, "kuzey koridor", fs)
    label(488, -200, "lokanta", fs); label(-500, -180, "berber", fs); label(-230, -120, "çay", fs)
    label(-660, 160, "emanet", fs); label(300, 272, "kuaför", fs); label(600, 160, "büfe", fs)
    label(-320, 216, "geçit", fs); label(820, 216, "geçit", fs)
    label(0, 700, "bozuk yürüyen merdivenler", fs); label(0, 1140, "KAPALI kapı", fs)
    label(-976, -600, "ara koridor", fs); label(976, -600, "ara koridor", fs)
im.save(dst)
print("yazıldı", dst)
