"""prev_plan.png üstüne bölge etiketleri yazar -> onizleme/plan.png
Kullanım: python plan_etiket.py [girdi] [cikti]
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "onizleme", "prev_plan.png")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "onizleme", "plan.png")
im = Image.open(src).convert("RGB")
W, H = im.size
# render: eye (0,-1,12000) -> (0,0,0), fov 21, 1600x1200 ; görünür yarı genişlik = tan(10.5°)*12000 ≈ 2224 birim
import math
half = math.tan(math.radians(10.5)) * 12000
sx = W / 2 / half
def P(x, y):
    return (W / 2 + x * sx, H / 2 - y * sx)
d = ImageDraw.Draw(im, "RGBA")
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
def label(x, y, text, font=f, fill=(255, 255, 255)):
    bb = d.textbbox((0, 0), text, font=font)
    px, py = P(x, y)
    px -= (bb[2] - bb[0]) / 2; py -= (bb[3] - bb[1]) / 2
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        d.text((px + ox, py + oy), text, font=font, fill=(0, 0, 0))
    d.text((px, py), text, font=font, fill=fill)
def zone(x0, y0, x1, y1, rgb):
    a, b = P(x0, y1), P(x1, y0)
    d.rectangle([a, b], fill=rgb + (70,), outline=rgb + (255,), width=3)
zone(740, -660, 1310, -220, (230, 60, 40)); label(1020, -440, "A", f2, (255, 90, 70))
zone(-1500, -680, -900, -220, (240, 200, 40)); label(-1200, -450, "B", f2, (255, 220, 60))
zone(80, -1500, 400, -1240, (230, 170, 40)); label(240, -1370, "T", f2, (255, 200, 60))
zone(-1420, 960, -1100, 1280, (60, 130, 240)); label(-1260, 1120, "CT", f2, (120, 170, 255))
label(-1090, 150, "Fabrika"); label(-1090, 90, "(çökmüş çatı, asma kat)", ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18))
label(0, -300, "Avlu"); label(0, 240, "Makine")
label(1024, 440, "Depo 2"); label(1024, -560, "Depo 1 (barut)")
label(1024, 0, "Ara sokak"); label(1440, -400, "Doğu sokağı")
label(500, 1150, "Çöp sahası"); label(800, 1430, "Baca + sarmal")
label(800, 820, "Yürüyüş yolu"); label(-700, 1300, "Yükleme alanı")
label(0, -900, "Yükleme yolu (T)"); label(0, -1620, "Ana kapı")
label(-400, 760, "Kuzey yolu")
im.save(dst)
print("yazıldı", dst)
